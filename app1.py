import os
import glob
import uuid
import time
import re
import urllib.parse
from datetime import datetime, timezone
import requests
import joblib
import pandas as pd
import streamlit as st
import google.generativeai as genai
from pymongo import MongoClient
import certifi
from sklearn.metrics.pairwise import cosine_similarity
from authlib.integrations.requests_client import OAuth2Session

import PyPDF2
import docx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_project_file(filename):
    direct_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(direct_path):
        return direct_path
    matches = glob.glob(os.path.join(BASE_DIR, "**", filename), recursive=True)
    return matches[0] if matches else direct_path

VEC_PATH = find_project_file("tfidf_vectorizer.pkl")
CLF_PATH = find_project_file("logistic_regression_validator.pkl")
SAMPLE_CSV_PATH = find_project_file("spark_sample_30.csv")

st.set_page_config(page_title="BugTriage-NLP", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------
# CUSTOM CSS FOR GREEN BUTTONS
# ---------------------------------------------------------
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Change Primary Buttons to Green */
        div.stButton > button[kind="primary"] {
            background-color: #198754 !important;
            color: white !important;
            border-color: #198754 !important;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #157347 !important;
            border-color: #157347 !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Standard secondary buttons styling */
        div.stButton > button[kind="secondary"] {
            border-radius: 6px;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# OAUTH 2.0 LOGIN LOGIC (GITHUB ONLY)
# ---------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

def handle_oauth_login():
    st.sidebar.markdown("### Authentication")
    
    if "github" not in st.secrets:
        st.sidebar.warning("GitHub OAuth secrets missing. Exporting disabled.")
        return

    if st.session_state.user is None:
        st.sidebar.info("Sign in to save history and export tickets.")
        query_params = st.query_params
        
        if "code" in query_params:
            code = query_params["code"]
            provider = query_params.get("state") 
            
            if provider == "github":
                try:
                    gh_cfg = st.secrets["github"]
                    client = OAuth2Session(
                        gh_cfg["client_id"], 
                        gh_cfg["client_secret"],
                        redirect_uri=gh_cfg["redirect_uri"]
                    )
                    token = client.fetch_token("https://github.com/login/oauth/access_token", code=code)
                    resp = client.get("https://api.github.com/user")
                    user_info = resp.json()
                    st.session_state.user = {
                        "name": user_info.get("name") or user_info.get("login"),
                        "email": user_info.get("email") or f"{user_info.get('login')}@github.user",
                        "provider": "GitHub",
                        "access_token": token.get("access_token")
                    }
                    st.query_params.clear()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"GitHub Login Failed: {e}")

        # Construct Authorization URL
        gh_cfg = st.secrets["github"]
        gh_auth_url = (
            f"https://github.com/login/oauth/authorize?"
            f"client_id={gh_cfg['client_id']}&"
            f"redirect_uri={urllib.parse.quote(gh_cfg['redirect_uri'])}&"
            f"scope=repo%20user:email&state=github"
        )

        st.sidebar.link_button("Login with GitHub", gh_auth_url, use_container_width=True)

    else:
        st.sidebar.success(f"Logged in as: **{st.session_state.user['name']}**")
        if st.sidebar.button("Log out", use_container_width=True):
            # Hard reset of the user session and the current active screen
            st.session_state.user = None
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            
            for key in ["current_report", "prediction", "confidence", "coherence_status", "coherence_reason", "draft_report"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.query_params.clear()
            st.rerun()

# Call the login handler
handle_oauth_login()

# ---------------------------------------------------------
# CORE SYSTEM INIT
# ---------------------------------------------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_resource
def load_chat_model():
    available_models = [
        m.name for m in genai.list_models() 
        if 'generateContent' in m.supported_generation_methods
    ]
    
    target_model = None
    for preferred in [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-latest"
    ]:
        if preferred in available_models:
            target_model = preferred
            break
            
    if not target_model:
        flash_models = [m for m in available_models if "flash" in m.lower()]
        target_model = flash_models[0] if flash_models else available_models[0]
        
    system_instruction = (
        "You are an expert Software Quality Assurance Engineer and Triage Specialist. "
        "Provide direct, concise, developer-ready outputs without conversational filler."
    )
    
    return genai.GenerativeModel(
        model_name=target_model,
        system_instruction=system_instruction
    )

@st.cache_resource
def load_local_ml_pipeline():
    vectorizer = joblib.load(VEC_PATH)
    validator = joblib.load(CLF_PATH)
    return vectorizer, validator

@st.cache_data
def load_sample_tickets():
    if os.path.exists(SAMPLE_CSV_PATH):
        try:
            return pd.read_csv(SAMPLE_CSV_PATH, encoding="utf-8")
        except Exception:
            return None
    return None

chat_model = load_chat_model()
vectorizer, validator = load_local_ml_pipeline()
sample_df = load_sample_tickets()

@st.cache_resource
def init_mongo():
    try:
        mongo_uri = st.secrets["MONGO_URI"]
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
        client.admin.command('ping')
        db = client["bugtriage_nlp_db"]
        
        db.sessions.create_index("session_id", unique=True)
        db.sessions.create_index("user_email")
        db.sessions.create_index([("timestamp", -1)])
        db.chat_history.create_index("session_id")
        db.chat_history.create_index([("timestamp", 1)])
        return db
    except Exception as e:
        st.sidebar.error(f"Failed to connect to MongoDB: {e}")
        return None

db = init_mongo()

def create_session(session_id, user_email, bug_report, prediction, confidence, coherence_status="COHERENT", coherence_reason=""):
    if db is None: return
    clean_text = bug_report.replace('\n', ' ').strip()
    title = clean_text[:35] + "..." if len(clean_text) > 35 else clean_text
    doc = {
        "session_id": session_id,
        "user_email": user_email,
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
    if db is None: return
    doc = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    }
    db.chat_history.insert_one(doc)

def get_all_sessions(user_email):
    if db is None: return []
    cursor = db.sessions.find({"user_email": user_email}, {"session_id": 1, "title": 1, "timestamp": 1, "_id": 0}).sort("timestamp", -1)
    return [(doc["session_id"], doc["title"], doc["timestamp"]) for doc in cursor]

def load_session(session_id):
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
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.draft_report = preserve_draft
    for key in ["current_report", "prediction", "confidence", "coherence_status", "coherence_reason"]:
        if key in st.session_state:
            del st.session_state[key]

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "draft_report" not in st.session_state:
    st.session_state.draft_report = ""

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

def evaluate_phase1_structure(report_text: str):
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

def local_heuristic_coherence_check(bug_report: str):
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', bug_report.lower())
    words = [w for w in clean_text.split() if len(w) > 2]
    
    if len(words) < 12:
        return "COHERENT", "Report too brief for split semantic analysis."

    midpoint = len(words) // 2
    vec1 = vectorizer.transform([" ".join(words[:midpoint])])
    vec2 = vectorizer.transform([" ".join(words[midpoint:])])
    similarity = cosine_similarity(vec1, vec2)[0][0]

    if similarity < 0.02:
        return "CONTRADICTORY", f"Low semantic continuity across report (similarity: {similarity:.3f})."

    return "COHERENT", f"Semantic continuity verified (similarity: {similarity:.3f})."

def verify_semantic_coherence(bug_report: str, max_retries=3):
    prompt = (
        "You are an automated Software Quality Assurance triage validator.\n\n"
        "Read the following software defect report and evaluate its semantic coherence:\n"
        f"\"\"\"{bug_report}\"\"\"\n\n"
        "Criteria for CONTRADICTORY:\n"
        "1. The report contains random, off-topic, or non-software sentences injected into the text "
        "(e.g., talking about animals, weather, food, or random daily life).\n"
        "2. The Expected Result and Actual Result logically contradict the Steps to Reproduce.\n\n"
        "Criteria for COHERENT:\n"
        "The entire report is a legitimate software issue description (normal greetings, sign-offs, "
        "logs, stack traces, and configuration snippets are valid).\n\n"
        "Respond in this exact 2-line format:\n"
        "VERDICT: [COHERENT or CONTRADICTORY]\n"
        "REASON: [One sentence explanation]"
    )
    
    for attempt in range(max_retries):
        try:
            response = chat_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=600
                ),
                request_options={"timeout": 15.0}
            )
            
            res = response.text.strip()
            upper_res = res.upper()
            
            is_contradictory = (
                "NON_SOFTWARE: YES" in upper_res
                or "NON_SOFTWARE_DETECTED: YES" in upper_res
                or "VERDICT: CONTRADICTORY" in upper_res
                or upper_res.startswith("CONTRADICTORY")
            )
            
            verdict = "CONTRADICTORY" if is_contradictory else "COHERENT"
            
            reason = ""
            for line in res.splitlines():
                if "REASON:" in line.upper():
                    reason = line.split(":", 1)[-1].strip()
                    break
            if not reason:
                clean_lines = [l.strip() for l in res.splitlines() if l.strip()]
                reason = " | ".join(clean_lines)[:140]
                
            return verdict, reason
        except Exception as e:
            if attempt == max_retries - 1:
                fb_verdict, fb_reason = local_heuristic_coherence_check(bug_report)
                return fb_verdict, f"[API Fallback ({type(e).__name__})] {fb_reason}"
            time.sleep(2)

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
        "When restructuring or analyzing, provide a thorough, complete explanation including:\n"
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
                    max_output_tokens=2048
                )
            )
            return response if stream else response.text
        except Exception as e:
            if attempt == max_retries - 1:
                return f"I encountered an error connecting to the cloud AI after {max_retries} retries: {e}"
            time.sleep(2)

def export_to_external_system(bug_report, standardized_body, prediction, confidence, coherence_status, coherence_reason, github_repo, access_token):
    clean_title = bug_report.strip().splitlines()[0][:65]
    issue_title = f"[BugTriage-NLP] {clean_title}"
    
    # -------------------------------------------------------------
    # NEW: Build the Validation Report header based on what has run
    # -------------------------------------------------------------
    val_text = "### 🤖 Automated Triage Validation\n"
    val_text += f"- **Phase 1 (Structural):** `{prediction}` ({confidence}% confidence)\n"
    
    # If Phase 2 was run, include it. If it's still PENDING, skip it.
    if coherence_status and coherence_status != "PENDING":
        val_text += f"- **Phase 2 (Semantic):** `{coherence_status}`\n"
        if coherence_reason:
            val_text += f"  - *Reason: {coherence_reason}*\n"
            
    val_text += "\n---\n\n"
    
    # Prepend the validation text to the AI ticket (or the raw report if no AI ticket exists)
    issue_body = val_text + (standardized_body if standardized_body else f"#### Raw Defect Description\n{bug_report}")

    if not access_token or not github_repo:
        return False, "Missing user access token or Repository name."

    url = f"https://api.github.com/repos/{github_repo}/issues"
    headers = {
        "Authorization": f"Bearer {access_token}",
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
            return True, f"Ticket successfully exported to: {issue_url}"
        return False, f"GitHub API Error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return False, f"Connection error: {e}"

# ---------------------------------------------------------
# UI RENDERING
# ---------------------------------------------------------

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
    
    # Hide History if not logged in
    if st.session_state.user is not None:
        st.header("Your Chat History")
        user_email = st.session_state.user["email"]
        sessions = get_all_sessions(user_email)

        if sessions:
            with st.container(height=400):
                for sess_id, title, timestamp in sessions:
                    date_str = timestamp.split(" ")[0]
                    if st.button(f"{title} ({date_str})", key=sess_id, use_container_width=True, help="Click to resume this triage session"):
                        load_session(sess_id)
                        st.rerun()
        else:
            st.info("No previous chats found for this account.")
    else:
        st.info("🔒 Sign in above to view and save your chat history.")

st.title("BugTriage-NLP")
st.markdown("**Automated Defect Report Validation and Triage Assistant**")
st.divider()

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
                final_prediction, valid_conf = evaluate_phase1_structure(clean_input)

                st.session_state.current_report = clean_input
                st.session_state.prediction = final_prediction
                st.session_state.confidence = valid_conf
                st.session_state.coherence_status = "PENDING"
                st.session_state.coherence_reason = ""

                # Save session linked to email (or 'anonymous' if guest)
                active_email = st.session_state.user["email"] if st.session_state.user else "anonymous"
                create_session(
                    st.session_state.session_id,
                    active_email,
                    clean_input,
                    st.session_state.prediction,
                    st.session_state.confidence,
                    st.session_state.coherence_status,
                    st.session_state.coherence_reason
                )
                st.rerun()
else:
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
        with coherence_col:
            st.caption("Phase 2: Semantic Coherence Verification")
            if st.session_state.get("coherence_status") == "PENDING":
                st.info("Awaiting Cloud Verification")
                if st.button("Run Coherence Check", type="secondary", use_container_width=True):
                    with st.spinner("Checking with Gemini (Will auto-retry if failed)..."):
                        verdict, reason = verify_semantic_coherence(st.session_state.current_report)
                        st.session_state.coherence_status = verdict
                        st.session_state.coherence_reason = reason
                        
                        active_email = st.session_state.user["email"] if st.session_state.user else "anonymous"
                        create_session(
                            st.session_state.session_id,
                            active_email,
                            st.session_state.current_report,
                            st.session_state.prediction,
                            st.session_state.confidence,
                            st.session_state.coherence_status,
                            st.session_state.coherence_reason
                        )
                        st.rerun()
            elif st.session_state.get("coherence_status") == "CONTRADICTORY":
                st.error("Flagged: Semantic Contradiction / Nonsense Detected")
                if st.session_state.get("coherence_reason"):
                    st.caption(f"Reason: {st.session_state.coherence_reason}")
            else:
                st.success("Semantic Check: Coherent")
                if st.session_state.get("coherence_reason") and "[API Fallback" in st.session_state.get("coherence_reason"):
                    st.caption(f"Note: {st.session_state.coherence_reason}")

        with action_col:
            st.caption("Phase 3: External Routing")
            
            if st.session_state.user is None:
                st.info("🔒 Sign in via the sidebar to export tickets.")
            else:
                target_repo = st.text_input("Target GitHub Repo", placeholder="owner/repo", help="Example: facebook/react")
                
                if st.button("Export to Tracking System", use_container_width=True, type="primary"):
                    if not target_repo.strip():
                        st.error("Please enter a target repository (owner/repo).")
                    else:
                        with st.spinner(f"Exporting to {target_repo}..."):
                            latest_ai_ticket = next(
                                (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
                                ""
                            )
                            # UPDATE: Pass all validation variables into the export function
                            success, msg_out = export_to_external_system(
                                bug_report=st.session_state.current_report,
                                standardized_body=latest_ai_ticket,
                                prediction=st.session_state.prediction,
                                confidence=st.session_state.get("confidence", 0.0),
                                coherence_status=st.session_state.get("coherence_status"),
                                coherence_reason=st.session_state.get("coherence_reason"),
                                github_repo=target_repo.strip(),
                                access_token=st.session_state.user.get("access_token")
                            )
                            if success:
                                st.toast("Export successful!")
                                st.success(msg_out)
                            else:
                                st.error(msg_out)

        st.markdown("### Triage Assistant")

        ai_col1, ai_col2, ai_col3 = st.columns([1, 1, 2])
        triggered_prompt = None

        with ai_col1:
            if st.button("Analyze Root Cause", use_container_width=True):
                triggered_prompt = "Analyze this report deeply, suggest the target developer component/team for routing, and identify the most probable root cause."

        with ai_col2:
            if st.button("Standardize Report Format", use_container_width=True):
                triggered_prompt = "Rewrite this bug report into a standardized developer-ready ticket including Severity Level, Target Developer Component/Routing, Steps to Reproduce, Expected vs. Actual Results, Flagged Contradictions, and Root-Cause Analysis."

        chat_container = st.container()
        with chat_container:
            for idx, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
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
            
            st.rerun()
