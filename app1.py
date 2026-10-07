import os
import glob
import uuid
import time
from datetime import datetime, timezone
import requests
import joblib
import pandas as pd
import streamlit as st
import google.generativeai as genai
from pymongo import MongoClient
import certifi

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

VEC_PATH = find_project_file("tfidf_vectorizer.pkl")
CLF_PATH = find_project_file("logistic_regression_validator.pkl")
SAMPLE_CSV_PATH = find_project_file("spark_sample_30.csv")

# --- PAGE CONFIG ---
st.set_page_config(page_title="BugTriage-NLP", layout="wide", initial_sidebar_state="expanded")

# --- API AND ML MODEL LOADING ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_resource
@st.cache_resource
def load_chat_model():
    """Dynamically finds the best available Gemini model for your specific API key."""
    # 1. Ask Google what models your key has permission to use
    available_models = [
        m.name for m in genai.list_models() 
        if 'generateContent' in m.supported_generation_methods
    ]
    
    # 2. Priority list: Target 3.8 Flash based on your dashboard limits
    target_model = None
    for preferred in ["models/gemini-3.8-flash", "models/gemini-3.8-flash-latest", "gemini-3.8-flash"]:
        if preferred in available_models:
            target_model = preferred
            break
            
    # 3. Ultimate fallback: Just grab the first available model if the preferences fail
    if not target_model and available_models:
        target_model = available_models[0]
        
    system_instruction = (
        "You are an expert Software Quality Assurance Engineer and Triage Specialist. "
        "Provide direct, concise, developer-ready outputs without conversational filler."
    )
    
    return genai.GenerativeModel(
        model_name=target_model,
        system_instruction=system_instruction
    )
    
    return genai.GenerativeModel(
        model_name=target_model,
        system_instruction=system_instruction
    )

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

# --- HELPER: EXACT VALID PROBABILITY & PHASE 1 EVALUATION ---
def evaluate_phase1_structure(report_text: str):
    """
    Calculates the exact probability of the report being 'Valid' (0% - 100%)
    and determines whether it passes Phase 1 or is flagged as 'Missing_Details'.
    """
    clean_input = report_text.strip()
    text_lower = clean_input.lower()
    word_count = len(clean_input.split())

    features = vectorizer.transform([clean_input])
    proba = validator.predict_proba(features)[0]
    classes = list(validator.classes_)

    if "Valid" in classes:
        valid_idx = classes.index("Valid")
    elif 1 in classes:
        valid_idx = classes.index(1)
    else:
        valid_idx = -1

    valid_conf = (proba[valid_idx] * 100) if valid_idx != -1 else (max(proba) * 100)
    valid_conf = round(float(valid_conf), 2)

    has_structure = (
        ("steps" in text_lower or "reproduce" in text_lower)
        and ("expected" in text_lower or "actual" in text_lower)
    )

    if word_count < 15:
        final_prediction = "Missing_Details"
    elif valid_conf >= 50.0 or (has_structure and valid_conf >= 40.0):
        final_prediction = "Valid"
    else:
        final_prediction = "Missing_Details"

    return final_prediction, valid_conf

# --- DATABASE SETUP & SESSION MANAGEMENT (MongoDB Persistence) ---
@st.cache_resource
def init_mongo():
    """Connects to MongoDB Atlas."""
    try:
        mongo_uri = st.secrets["MONGO_URI"]
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
        client.admin.command('ping')
        db = client["bugtriage_nlp_db"]
        
        db.sessions.create_index("session_id", unique=True)
        db.sessions.create_index([("timestamp", -1)])
        db.chat_history.create_index("session_id")
        db.chat_history.create_index([("timestamp", 1)])
        return db
    except Exception as e:
        st.sidebar.error(f"Failed to connect to MongoDB: {e}")
        return None

db = init_mongo()

def create_session(session_id, bug_report, prediction, confidence, coherence_status="COHERENT", coherence_reason=""):
    """Creates or updates a session document in MongoDB."""
    if db is None: return
    clean_text = bug_report.replace('\n', ' ').strip()
    title = clean_text[:35] + "..." if len(clean_text) > 35 else clean_text

    doc = {
        "session_id": session_id,
        "title": title,
        "bug_report": bug_report,
        "prediction": prediction,
        "confidence": confidence,
        "coherence_status": coherence_status,
        "coherence_reason": coherence_reason,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    }
    db.sessions.update_one({"session_id": session_id}, {"$set": doc}, upsert=True)

def save_message(session_id, role, content):
    """Saves a single chat message document to MongoDB."""
    if db is None: return
    doc = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    }
    db.chat_history.insert_one(doc)

def get_all_sessions():
    """Retrieves all sessions for the sidebar history."""
    if db is None: return []
    cursor = db.sessions.find({}, {"session_id": 1, "title": 1, "timestamp": 1, "_id": 0}).sort("timestamp", -1)
    return [(doc["session_id"], doc["title"], doc["timestamp"]) for doc in cursor]

def load_session(session_id):
    """Loads a previous session's exact metrics and messages from MongoDB into session_state."""
    if db is None: return
    session_data = db.sessions.find_one({"session_id": session_id})

    if session_data:
        bug_report = session_data.get("bug_report", "")
        saved_pred = session_data.get("prediction", "")
        saved_conf = session_data.get("confidence")
        coh_status = session_data.get("coherence_status", "COHERENT")
        coh_reason = session_data.get("coherence_reason", "")

        if saved_conf is None:
            recalc_pred, recalc_conf = evaluate_phase1_structure(bug_report)
            saved_conf = recalc_conf
            saved_pred = saved_pred or recalc_pred

        st.session_state.session_id = session_id
        st.session_state.current_report = bug_report
        st.session_state.prediction = saved_pred
        st.session_state.confidence = round(float(saved_conf), 2)
        st.session_state.coherence_status = coh_status
        st.session_state.coherence_reason = coh_reason

        messages_cursor = db.chat_history.find({"session_id": session_id}).sort("timestamp", 1)
        st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages_cursor]

def start_new_chat(preserve_draft=""):
    """Resets the application state for a new or revised bug report."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.draft_report = preserve_draft
    for key in ["current_report", "prediction", "confidence", "coherence_status", "coherence_reason"]:
        if key in st.session_state:
            del st.session_state[key]

# Session tracking Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "draft_report" not in st.session_state:
    st.session_state.draft_report = ""

# --- TEXT EXTRACTION FUNCTION ---
@st.cache_data(show_spinner=False)
def extract_text_from_file(file_bytes: bytes, file_name: str):
    import io
    file_extension = file_name.split('.')[-1].lower()
    extracted_text = ""
    try:
        if file_extension == 'txt':
            extracted_text = file_bytes.decode("utf-8", errors="ignore")
        elif file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                extracted_text += (page.extract_text() or "") + "\n"
        elif file_extension == 'docx':
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading the file: {e}")
    return extracted_text

# --- PHASE 2: FAST SEMANTIC COHERENCE VERIFICATION (WITH RETRIES) ---
# --- PHASE 2: FAST SEMANTIC COHERENCE VERIFICATION (WITH RETRIES) ---
def verify_semantic_coherence(bug_report: str, max_retries=3):
    """Fast, token-capped check for logical alignment between Steps and Results with auto-retry."""
    prompt = (
        "Check if this bug report has a severe logical contradiction between the Steps to Reproduce "
        "and the Expected/Actual Results, OR if it contains nonsensical/irrelevant statements "
        "(e.g., mentioning weather, food, or impossible events).\n\n"
        f"Bug Report:\n\"\"\"{bug_report}\"\"\"\n\n"
        "Respond in EXACTLY two lines:\n"
        "VERDICT: [COHERENT or CONTRADICTORY]\n"
        "REASON: [One short sentence]"
    )
    
    for attempt in range(max_retries):
        try:
            res = chat_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=60
                )
            ).text.strip()
            lines = [line.strip() for line in res.splitlines() if line.strip()]
            verdict = "CONTRADICTORY" if lines and "CONTRADICTORY" in lines[0].upper() else "COHERENT"
            reason = lines[-1].replace("REASON:", "").strip() if len(lines) > 1 else ""
            return verdict, reason
        except Exception as e:
            if attempt == max_retries - 1:
                return "COHERENT", f"Coherence check bypassed after {max_retries} retries ({e})"
            time.sleep(2)

# --- PHASE 2: OPTIMIZED GENERATIVE AI RESTRUCTURING (WITH RETRIES) ---
def generate_ai_response(user_question, bug_report_context, prediction_status, guideline_text="", stream=True, max_retries=3):
    if not bug_report_context or not bug_report_context.strip():
        return "I do not have a bug report to look at yet. Please analyze one first."

    current_date = datetime.now().strftime('%Y-%m-%d')
    guideline_block = f"\nQA Formatting Guidelines:\n{guideline_text}\n" if guideline_text else ""

    prompt = (
        f"Date: {current_date}\n"
        f"Phase 1 Validation Status: {prediction_status}\n"
        f"{guideline_block}\n"
        f"Raw Bug Report:\n\"\"\"{bug_report_context}\"\"\"\n\n"
        f"Task: {user_question}\n\n"
        "When restructuring or analyzing, include:\n"
        "1. **Standardized Title & Summary**\n"
        "2. **Severity Level** (Blocker, Critical, Major, Minor, Trivial)\n"
        "3. **Component Routing** (Backend/Core, SQL/Database, UI/Frontend, Python/API, Network/Cluster)\n"
        "4. **Steps to Reproduce, Expected Result, & Actual Result**\n"
        "5. **Flagged Contradictions / Ambiguities**\n"
        "6. **Probable Root-Cause Hypothesis**"
    )
    
    for attempt in range(max_retries):
        try:
            response = chat_model.generate_content(
                prompt,
                stream=stream,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=900
                )
            )
            return response if stream else response.text
        except Exception as e:
            if attempt == max_retries - 1:
                return f"I encountered an error connecting to the cloud AI after {max_retries} retries: {e}"
            time.sleep(2)

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
        custom_guideline_text = extract_text_from_file(uploaded_guideline.getvalue(), uploaded_guideline.name)
        if custom_guideline_text.strip():
            st.success("Guidelines active.")

    st.divider()

    st.header("Chat History")
    sessions = get_all_sessions()

    if sessions:
        with st.container(height=400):
            for sess_id, title, timestamp in sessions:
                date_str = timestamp.split(" ")[0]
                if st.button(f"{title} ({date_str})", key=sess_id, use_container_width=True, help="Click to resume this triage session"):
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

                # Phase 1: Local TF-IDF + Logistic Regression Evaluation (INSTANT!)
                final_prediction, valid_conf = evaluate_phase1_structure(clean_input)

                st.session_state.current_report = clean_input
                st.session_state.prediction = final_prediction
                st.session_state.confidence = valid_conf
                
                # Defer the Gemini call so Phase 1 is lightning-fast
                st.session_state.coherence_status = "PENDING"
                st.session_state.coherence_reason = ""

                create_session(
                    st.session_state.session_id,
                    clean_input,
                    st.session_state.prediction,
                    st.session_state.confidence,
                    st.session_state.coherence_status,
                    st.session_state.coherence_reason
                )
                st.rerun()
else:
    # Display the active report context
    with st.expander("View Active Defect Report Context", expanded=False):
        st.write(st.session_state.current_report)

    status_col, coherence_col, action_col = st.columns([1.2, 1.2, 1])
    with status_col:
        st.caption("Phase 1: Local Validation (Logistic Regression)")
        conf_val = st.session_state.get("confidence", 0.0)
        conf_str = f" ({conf_val:.2f}% conf)"
        if st.session_state.prediction == "Valid":
            st.success(f"Status: Valid Structure{conf_str}")
        else:
            st.error(f"Status: Missing Details{conf_str}")

    # --- BLOCK PHASE 2 CLOUD CALLS WHEN FLAGGED AS MISSING_DETAILS ---
    if st.session_state.prediction == "Missing_Details":
        with coherence_col:
            st.caption("Phase 2: Cloud LLM Gatekeeper")
            st.warning("Cloud AI Skipped (Blocked by Phase 1)")

        st.warning(
            f"**Revision Required (`Missing_Details` — {conf_val:.2f}% Valid Score):** This defect report scored below the "
            "structural validity threshold. To conserve cloud resources, generative AI processing is withheld. "
            "Please revise the report to include at least 15 words with clear **Steps to Reproduce**, **Expected Result**, and **Actual Result**."
        )
        if st.button("Revise & Resubmit Report", type="primary"):
            start_new_chat(preserve_draft=st.session_state.current_report)
            st.rerun()

    else:
        # Report is "Valid" -> Show Manual Trigger for Phase 2, Triage Assistant, & Phase 3 Export
        with coherence_col:
            st.caption("Phase 2: Semantic Coherence Verification")
            if st.session_state.get("coherence_status") == "PENDING":
                st.info("Awaiting Cloud Verification")
                if st.button("Run Coherence Check", type="secondary", use_container_width=True):
                    with st.spinner("Checking with Gemini (Will auto-retry if failed)..."):
                        verdict, reason = verify_semantic_coherence(st.session_state.current_report)
                        st.session_state.coherence_status = verdict
                        st.session_state.coherence_reason = reason
                        
                        create_session(
                            st.session_state.session_id,
                            st.session_state.current_report,
                            st.session_state.prediction,
                            st.session_state.confidence,
                            st.session_state.coherence_status,
                            st.session_state.coherence_reason
                        )
                        st.rerun()
            elif st.session_state.get("coherence_status") == "CONTRADICTORY":
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

        # Quick-action buttons trigger instant streaming instead of blocking spinners
        ai_col1, ai_col2, ai_col3 = st.columns([1, 1, 2])
        triggered_prompt = None

        with ai_col1:
            if st.button("Analyze Root Cause", use_container_width=True):
                triggered_prompt = "Analyze this report deeply, suggest the target developer component/team for routing, and identify the most probable root cause."

        with ai_col2:
            if st.button("Standardize Report Format", use_container_width=True):
                triggered_prompt = "Rewrite this bug report into a standardized developer-ready ticket including Severity Level, Target Developer Component/Routing, Steps to Reproduce, Expected vs. Actual Results, Flagged Contradictions, and Root-Cause Analysis."

        # Render existing conversation history
        chat_container = st.container()
        with chat_container:
            for idx, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # AUTO-EXPORT & COPY BUTTONS for Assistant Messages
                    if message["role"] == "assistant":
                        dl_col, copy_col, space = st.columns([2, 3, 5])
                        with dl_col:
                            st.download_button(
                                label="📥 Export (.md)",
                                data=message["content"],
                                file_name=f"BugTriage_Report_{idx}.md",
                                mime="text/markdown",
                                key=f"dl_hist_{idx}"
                            )
                        with copy_col:
                            with st.expander("📋 View/Copy Raw Code"):
                                st.code(message["content"], language="markdown")

        # Handle either a quick-action button click or chat_input submission with live streaming
        chat_box_input = st.chat_input("Enter a query regarding this report...")
        active_query = triggered_prompt or chat_box_input

        if active_query:
            st.session_state.messages.append({"role": "user", "content": active_query})
            save_message(st.session_state.session_id, "user", active_query)

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(active_query)

                with st.chat_message("assistant"):
                    response_stream = generate_ai_response(
                        user_question=active_query,
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
                                if chunk.text:
                                    full_text += chunk.text
                                    yield chunk.text
                            save_message(st.session_state.session_id, "assistant", full_text)

                        full_reply = st.write_stream(stream_text)
                        st.session_state.messages.append({"role": "assistant", "content": full_reply})
            
            # Force rerun so the newly generated message gets the Export & Copy buttons
            st.rerun()
