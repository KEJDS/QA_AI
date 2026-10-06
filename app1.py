import os
import streamlit as st
import joblib
import pandas as pd
import google.generativeai as genai
import requests 
from datetime import datetime
import sqlite3
import uuid
import glob

# --- IMPORTS FOR FILE PARSING ---
import PyPDF2
import docx

# --- BASE DIRECTORY (Works locally and on GitHub/Streamlit Cloud) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_project_file(filename):
    """Searches BASE_DIR and all subfolders for the target file."""
    direct_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(direct_path):
        return direct_path
    matches = glob.glob(os.path.join(BASE_DIR, "**", filename), recursive=True)
    return matches[0] if matches else direct_path

DB_PATH = os.path.join(BASE_DIR, "chat_logs.db")
VEC_PATH = find_project_file("tfidf_vectorizer.pkl")
CLF_PATH = find_project_file("logistic_regression_validator.pkl")
SAMPLE_CSV_PATH = find_project_file("spark_sample_30.csv")

# --- PAGE CONFIG ---
st.set_page_config(page_title="BugTriage-NLP", layout="wide", initial_sidebar_state="expanded")

# --- DATABASE SETUP & SESSION MANAGEMENT (SQLite Temporary Persistence) ---
def init_db():
    """Initializes the SQLite database with Sessions and ChatHistory tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            bug_report TEXT,
            prediction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ChatHistory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def create_session(session_id, bug_report, prediction):
    """Creates a new session record with an auto-generated title."""
    clean_text = bug_report.replace('\n', ' ').strip()
    title = clean_text[:35] + "..." if len(clean_text) > 35 else clean_text
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO Sessions (session_id, title, bug_report, prediction) VALUES (?, ?, ?, ?)", 
        (session_id, title, bug_report, prediction)
    )
    conn.commit()
    conn.close()

def save_message(session_id, role, content):
    """Saves a single chat message to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO ChatHistory (session_id, role, content) VALUES (?, ?, ?)", 
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def get_all_sessions():
    """Retrieves all sessions for the sidebar history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, title, timestamp FROM Sessions ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()
    return records

def load_session(session_id):
    """Loads a previous session's context and messages into the active state."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT bug_report, prediction FROM Sessions WHERE session_id = ?", (session_id,))
    session_data = cursor.fetchone()
    
    if session_data:
        st.session_state.session_id = session_id
        st.session_state.current_report = session_data[0]
        st.session_state.prediction = session_data[1]
        
        cursor.execute("SELECT role, content FROM ChatHistory WHERE session_id = ? ORDER BY id ASC", (session_id,))
        messages = cursor.fetchall()
        st.session_state.messages = [{"role": row[0], "content": row[1]} for row in messages]
        
    conn.close()

def start_new_chat(preserve_draft=""):
    """Resets the application state for a new or revised bug report."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.draft_report = preserve_draft
    for key in ["current_report", "prediction", "confidence", "coherence_status", "coherence_reason"]:
        if key in st.session_state:
            del st.session_state[key]

# Initialize database on app startup
init_db()

# Session tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "draft_report" not in st.session_state:
    st.session_state.draft_report = ""

# --- API AND ML MODEL LOADING ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_resource
def load_chat_model():
    # Uses gemini-2.5-flash by default (or override via GEMINI_MODEL in secrets.toml)
    model_name = st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash")
    return genai.GenerativeModel(model_name)

@st.cache_resource
def load_local_ml_pipeline():
    """Loads the Phase 1 TF-IDF Vectorizer and Logistic Regression Classifier."""
    vectorizer = joblib.load(VEC_PATH)
    validator = joblib.load(CLF_PATH)
    return vectorizer, validator

@st.cache_data
def load_sample_tickets():
    """Safely loads the 30 sample Spark tickets if a valid CSV is present."""
    if os.path.exists(SAMPLE_CSV_PATH):
        try:
            return pd.read_csv(SAMPLE_CSV_PATH, encoding="utf-8")
        except Exception:
            return None
    return None

chat_model = load_chat_model()
vectorizer, validator = load_local_ml_pipeline()
sample_df = load_sample_tickets()

# --- TEXT EXTRACTION FUNCTION ---
def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    extracted_text = ""
    try:
        if file_extension == 'txt':
            extracted_text = str(uploaded_file.read(), "utf-8")
        elif file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted_text += (page.extract_text() or "") + "\n"
        elif file_extension == 'docx':
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading the file: {e}")
    return extracted_text

# --- PHASE 2: SEMANTIC COHERENCE VERIFICATION ---
def verify_semantic_coherence(bug_report):
    """Checks if Steps to Reproduce and Expected/Actual Results logically align."""
    prompt = f"""
    You are a Software Quality Assurance Gatekeeper performing Semantic Coherence Verification.
    Inspect the following bug report and determine if there is a severe logical contradiction 
    between the Steps to Reproduce and the Expected/Actual Results (for example, steps describe 
    logging in, but the actual result complains about printer hardware or unrelated features).

    Bug Report:
    "{bug_report}"

    Respond in EXACTLY this two-line format:
    VERDICT: [COHERENT or CONTRADICTORY]
    REASON: [One concise sentence explaining why]
    """
    try:
        res = chat_model.generate_content(prompt).text.strip()
        verdict = "CONTRADICTORY" if "CONTRADICTORY" in res.upper().splitlines()[0] else "COHERENT"
        reason = res.splitlines()[-1].replace("REASON:", "").strip() if len(res.splitlines()) > 1 else ""
        return verdict, reason
    except Exception as e:
        return "COHERENT", f"Coherence check bypassed ({e})"

# --- PHASE 2: GENERATIVE AI RESTRUCTURING & TRIAGE ---
def generate_ai_response(user_question, bug_report_context, prediction_status, guideline_text="", stream=False):
    if not bug_report_context or bug_report_context.strip() == "":
        return "I do not have a bug report to look at yet. Please analyze one first."
        
    current_date = datetime.now().strftime('%Y-%m-%d')
    guideline_instructions = (
        f"\nCRITICAL INSTRUCTION: Strictly format and evaluate the bug report against these uploaded QA guidelines:\n{guideline_text}\n"
        if guideline_text else ""
    )
    
    prompt = f"""
    You are an expert Software Quality Assurance Engineer and Triage Specialist assisting an Agile software development team.
    System Context: Today's date is {current_date}. {guideline_instructions}
    
    You are reviewing the following raw bug report:
    "{bug_report_context}"
    
    The Phase 1 local Scikit-Learn Logistic Regression classifier flagged this report as: {prediction_status}.
    
    The user is asking you this request: "{user_question}"
    
    Answer the user's request directly, clearly, and practically.
    When restructuring or analyzing the report, include:
    1. Standardized Title & Summary
    2. Identified Severity Level (Blocker, Critical, Major, Minor, or Trivial)
    3. Recommended Developer Routing / Component Assignment (e.g., Backend/Core, SQL/Database, UI/Frontend, Python/API, or Network/Cluster)
    4. Structured Steps to Reproduce, Expected Result, and Actual Result
    5. Flagged Contradictions / Ambiguities (if any exist between steps and results)
    6. AI-Assisted Probable Root-Cause Hypothesis
    """
    try:
        response = chat_model.generate_content(prompt, stream=stream)
        return response if stream else response.text
    except Exception as e:
        return f"I encountered an error connecting to the cloud AI: {e}"

# --- PHASE 3: GITHUB REST API ISSUE EXPORT ---
def export_to_external_system(bug_report, standardized_body, prediction):
    """Exports the validated and restructured ticket directly to GitHub Issues via REST API."""
    github_token = st.secrets.get("GITHUB_TOKEN", "")
    github_repo = st.secrets.get("GITHUB_REPO", "")

    clean_title = bug_report.strip().splitlines()[0][:65]
    issue_title = f"[BugTriage-NLP] {clean_title}"
    issue_body = standardized_body if standardized_body else (
        f"### Validated Defect Report\n**Phase 1 Status:** `{prediction}`\n\n"
        f"#### Raw Defect Description\n{bug_report}"
    )

    if not github_token or not github_repo:
        return True, "Ticket exported (Add GITHUB_TOKEN and GITHUB_REPO in secrets.toml for live GitHub repository issue creation)."

    url = f"https://api.github.com/repos/{github_repo}/issues"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json"
    }
    payload = {
        "title": issue_title,
        "body": issue_body,
        "labels": ["bug", "triage-validated"]
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code in (200, 201):
            issue_url = resp.json().get("html_url", "")
            return True, f"Ticket successfully exported to GitHub Issues: {issue_url}"
        return False, f"GitHub API Error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return False, f"Connection error: {e}"

# --- STREAMLIT UI STYLING ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stButton>button {
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR PANELS ---
with st.sidebar:
    if st.button("New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()
        
    st.divider()
    
    st.header("Document Uploads")
    st.write("Upload QA formatting guidelines (.txt, .pdf, .docx).")
    uploaded_guideline = st.file_uploader("Select File", type=["txt", "pdf", "docx"], label_visibility="collapsed")
    custom_guideline_text = ""
    if uploaded_guideline is not None:
        with st.spinner("Processing document..."):
            custom_guideline_text = extract_text_from_file(uploaded_guideline)
        if custom_guideline_text.strip():
            st.success("Guidelines active.")
            
    st.divider()
    
    st.header("Chat History")
    sessions = get_all_sessions()
    
    if sessions:
        with st.container(height=400):
            for sess_id, title, timestamp in sessions:
                date_str = timestamp.split(" ")[0] 
                if st.button(f"{title} ({date_str})", key=sess_id, help="Click to resume this triage session"):
                    load_session(sess_id)
                    st.rerun()
    else:
        st.info("No previous chats found.")

# --- MAIN APPLICATION ---
st.title("BugTriage-NLP")
st.markdown("**Automated Defect Report Validation and Triage Assistant**")
st.divider()

# Input box when starting a fresh chat or revising an incomplete report
if "current_report" not in st.session_state:
    st.write("Input the defect description below. The local Scikit-Learn Logistic Regression model will validate its structural integrity, followed by Gemini AI refinement.")
    
    default_text = st.session_state.get("draft_report", "")
    if sample_df is not None and not sample_df.empty:
        id_col = next((c for c in sample_df.columns if 'id' in c.lower()), sample_df.columns[0])
        desc_col = 'Description_Clean' if 'Description_Clean' in sample_df.columns else next((c for c in sample_df.columns if 'desc' in c.lower()), sample_df.columns[-1])
        
        options = ["-- Type manually below --"] + [
            f"Ticket #{i+1}: {row[id_col]}" for i, row in sample_df.iterrows()
        ]
        selected_option = st.selectbox("Load a sample historical ticket from GitBugs (Optional):", options)
        if selected_option != "-- Type manually below --":
            row_idx = int(selected_option.split("#")[1].split(":")[0]) - 1
            default_text = str(sample_df.iloc[row_idx][desc_col])

    user_input = st.text_area(
        "Defect Description",
        value=default_text,
        height=150,
        placeholder="Enter bug report details (Steps to Reproduce, Expected Result, Actual Result)...",
        label_visibility="collapsed"
    )

    action_col1, action_col2, action_col3 = st.columns([1, 1, 2])

    with action_col1:
        if st.button("Analyze Report", use_container_width=True, type="primary"):
            if user_input.strip() == "":
                st.warning("Please enter a defect description before proceeding.")
            else:
                clean_input = user_input.strip()
                text_lower = clean_input.lower()
                word_count = len(clean_input.split())
                
                # Phase 1: TF-IDF Vectorization + Logistic Regression Prediction
                features = vectorizer.transform([clean_input])
                ml_prediction = validator.predict(features)[0]
                confidence = max(validator.predict_proba(features)[0]) * 100
                
                # Structural keyword check + minimum length gatekeeper
                has_structure = (
                    ("steps" in text_lower or "reproduce" in text_lower)
                    and ("expected" in text_lower or "actual" in text_lower)
                )
                
                if word_count < 15:
                    final_prediction = "Missing_Details"
                elif has_structure or ml_prediction == "Valid":
                    final_prediction = "Valid"
                else:
                    final_prediction = "Missing_Details"
                
                # Save state
                st.session_state.current_report = clean_input
                st.session_state.prediction = final_prediction
                st.session_state.confidence = round(confidence, 2)

                # Phase 2 Semantic Coherence Check (Only executed if Phase 1 == "Valid")
                if final_prediction == "Valid":
                    with st.spinner("Phase 2: Checking semantic coherence with Gemini AI..."):
                        verdict, reason = verify_semantic_coherence(clean_input)
                        st.session_state.coherence_status = verdict
                        st.session_state.coherence_reason = reason
                
                # Log session in database
                create_session(st.session_state.session_id, clean_input, st.session_state.prediction)
                st.rerun()
else:
    # Display the active report context
    with st.expander("View Active Defect Report Context", expanded=False):
        st.write(st.session_state.current_report)
        
    status_col, coherence_col, action_col = st.columns([1.2, 1.2, 1])
    with status_col:
        st.caption("Phase 1: Local Validation (Logistic Regression)")
        conf_str = f" ({st.session_state.confidence}% conf)" if "confidence" in st.session_state else ""
        if st.session_state.prediction == "Valid":
            st.info(f"Status: Valid Structure{conf_str}")
        else:
            st.error(f"Status: Missing Details{conf_str}")

    # --- BLOCK PHASE 2 CLOUD CALLS WHEN FLAGGED AS MISSING_DETAILS (Matches Figure 3) ---
    if st.session_state.prediction == "Missing_Details":
        with coherence_col:
            st.caption("Phase 2: Cloud LLM Gatekeeper")
            st.warning("Cloud AI Skipped (Blocked by Phase 1)")

        st.warning(
            "**Revision Required (`Missing_Details`):** This defect report did not pass Phase 1 local structural validation. "
            "To conserve cloud resources, generative AI processing is withheld. Please revise the report to include at least "
            "15 words with clear **Steps to Reproduce**, **Expected Result**, and **Actual Result**."
        )
        if st.button("Revise & Resubmit Report", type="primary"):
            start_new_chat(preserve_draft=st.session_state.current_report)
            st.rerun()

    else:
        # Report is "Valid" -> Show Phase 2 Coherence Verdict, Triage Assistant, & Phase 3 Export
        with coherence_col:
            st.caption("Phase 2: Semantic Coherence Verification")
            if st.session_state.get("coherence_status") == "CONTRADICTORY":
                st.warning("Flagged: Logical Contradiction Detected")
                if st.session_state.get("coherence_reason"):
                    st.caption(f"Reason: {st.session_state.coherence_reason}")
            else:
                st.success("Semantic Check: Coherent")

        with action_col:
            st.caption("Phase 3: External Routing")
            if st.button("Export to Tracking System", use_container_width=True):
                with st.spinner("Connecting to GitHub REST API..."):
                    latest_ai_ticket = next(
                        (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
                        ""
                    )
                    success, msg_out = export_to_external_system(
                        st.session_state.current_report,
                        latest_ai_ticket,
                        st.session_state.prediction
                    )
                    if success:
                        st.toast(msg_out)
                        st.success(msg_out)
                    else:
                        st.error(msg_out)

        st.markdown("### Triage Assistant")
        
        ai_col1, ai_col2, ai_col3 = st.columns([1, 1, 2])
        with ai_col1:
            if st.button("Analyze Root Cause", use_container_width=True):
                msg = "Analyze this report deeply, suggest the target developer component/team for routing, and identify the most probable root cause."
                st.session_state.messages.append({"role": "user", "content": msg})
                save_message(st.session_state.session_id, "user", msg)
                
                with st.spinner("Processing analysis..."):
                    reply = generate_ai_response(
                        msg,
                        st.session_state.current_report,
                        st.session_state.prediction,
                        guideline_text=custom_guideline_text,
                        stream=False
                    )
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    save_message(st.session_state.session_id, "assistant", reply)
                    st.rerun()

        with ai_col2:
            if st.button("Standardize Report Format", use_container_width=True):
                msg = "Rewrite this bug report into a standardized developer-ready ticket including Severity Level, Target Developer Component/Routing, Steps to Reproduce, Expected vs. Actual Results, Flagged Contradictions, and Root-Cause Analysis."
                st.session_state.messages.append({"role": "user", "content": msg})
                save_message(st.session_state.session_id, "user", msg)
                
                with st.spinner("Restructuring..."):
                    reply = generate_ai_response(
                        msg,
                        st.session_state.current_report,
                        st.session_state.prediction,
                        guideline_text=custom_guideline_text,
                        stream=False
                    )
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    save_message(st.session_state.session_id, "assistant", reply)
                    st.rerun()

        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt_input := st.chat_input("Enter a query regarding this report..."):
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            save_message(st.session_state.session_id, "user", prompt_input)
            
            with st.chat_message("user"):
                st.markdown(prompt_input)

            with st.chat_message("assistant"):
                response_stream = generate_ai_response(
                    user_question=prompt_input, 
                    bug_report_context=st.session_state.current_report,
                    prediction_status=st.session_state.prediction,
                    guideline_text=custom_guideline_text,
                    stream=True
                )
                
                if isinstance(response_stream, str):
                    st.markdown(response_stream)
                    st.session_state.messages.append({"role": "assistant", "content": response_stream})
                    save_message(st.session_state.session_id, "assistant", response_stream)
                else:
                    def stream_text():
                        full_text = ""
                        for chunk in response_stream:
                            full_text += chunk.text
                            yield chunk.text
                        save_message(st.session_state.session_id, "assistant", full_text)
                    
                    full_reply = st.write_stream(stream_text)
                    st.session_state.messages.append({"role": "assistant", "content": full_reply})
