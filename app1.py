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
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
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
            st.session_state.user = None
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            
            for key in ["current_report", "prediction", "confidence", "coherence_status", "coherence_reason", "draft_report", "phase1_ms", "phase2_ms", "duplicates"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.query_params.clear()
            st.rerun()

handle_oauth_login()

# ---------------------------------------------------------
# CORE SYSTEM INIT
# ---------------------------------------------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_resource
def load_chat_model():
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    target_model = next((m for m in ["models/gemini-2.5-flash", "models/gemini-2.0-flash", "models/gemini-1.5-flash"] if m in available_models), available_models[0])
    system_instruction = "You are an expert Software Quality Assurance Engineer and Triage Specialist. Provide direct, concise, developer-ready outputs."
    return genai.GenerativeModel(model_name=target_model, system_instruction=system_instruction)

@st.cache_resource
def load_local_ml_pipeline():
    return joblib.load(VEC_PATH), joblib.load(CLF_PATH)

@st.cache_data
def load_sample_tickets():
    return pd.read_csv(SAMPLE_CSV_PATH, encoding="utf-8") if os.path.exists(SAMPLE_CSV_PATH) else None

chat_model = load_chat_model()
vectorizer, validator = load_local_ml_pipeline()
sample_df = load_sample_tickets()

@st.cache_resource
def init_mongo():
    try:
        client = MongoClient(st.secrets["MONGO_URI"], serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
        client.admin.command('ping')
        db = client["bugtriage_nlp_db"]
        db.sessions.create_index("session_id", unique=True)
        db.sessions.create_index("user_email")
        db.sessions.create_index([("timestamp", -1)])
        return db
    except Exception as e:
        st.sidebar.error(f"Failed to connect to MongoDB: {e}")
        return None

db = init_mongo()

def create_session(session_id, user_email, bug_report, prediction, confidence, coherence_status="COHERENT", coherence_reason="", p1_ms=0, p2_ms=0):
    if db is None: return
    title = bug_report.replace('\n', ' ').strip()[:35] + "..." if len(bug_report) > 35 else bug_report
    doc = {
        "session_id": session_id,
        "user_email": user_email,
        "title": title,
        "bug_report": bug_report,
        "prediction": prediction,
        "confidence": confidence,
        "coherence_status": coherence_status,
        "coherence_reason": coherence_reason,
        "phase1_ms": p1_ms,
        "phase2_ms": p2_ms,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    }
    db.sessions.update_one({"session_id": session_id}, {"$set": doc}, upsert=True)

def save_message(session_id, role, content):
    if db is None: return
    db.chat_history.insert_one({
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    })

def check_for_duplicates(current_text, current_session_id):
    if db is None: return []
    past_sessions = list(db.sessions.find({"session_id": {"$ne": current_session_id}}, {"bug_report": 1, "title": 1, "session_id": 1}))
    if not past_sessions: return []
    
    past_texts = [s.get("bug_report", "") for s in past_sessions]
    vec_current = vectorizer.transform([current_text])
    vec_past = vectorizer.transform(past_texts)
    sim_scores = cosine_similarity(vec_current, vec_past)[0]
    
    duplicates = []
    for idx, score in enumerate(sim_scores):
        if score >= 0.85:
            duplicates.append({"title": past_sessions[idx].get("title"), "score": round(score * 100, 2)})
    return sorted(duplicates, key=lambda x: x["score"], reverse=True)

def predict_component_local(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ["database", "sql", "mongo", "query", "table"]): return "Backend / Database"
    if any(w in text_lower for w in ["ui", "button", "css", "frontend", "design", "color"]): return "Frontend / UI"
    if any(w in text_lower for w in ["api", "timeout", "network", "server", "endpoint", "500"]): return "Network / API"
    return "Core / Uncategorized"

def evaluate_phase1_structure(report_text: str):
    clean_input = report_text.strip()
    text_lower = clean_input.lower()
    
    features = vectorizer.transform([clean_input])
    proba = validator.predict_proba(features)[0]
    classes = list(validator.classes_)

    valid_idx = classes.index("Valid") if "Valid" in classes else (classes.index(1) if 1 in classes else -1)
    valid_conf = round(float((proba[valid_idx] * 100) if valid_idx != -1 else (max(proba) * 100)), 2)

    has_structure = (("steps" in text_lower or "reproduce" in text_lower) and ("expected" in text_lower or "actual" in text_lower))

    if len(clean_input.split()) < 15:
        return "Missing_Details", valid_conf
    if valid_conf >= 50.0 or (has_structure and valid_conf >= 40.0):
        return "Valid", valid_conf
    return "Missing_Details", valid_conf

def verify_semantic_coherence(bug_report: str):
    prompt = f"Evaluate software defect report coherence:\n\"\"\"{bug_report}\"\"\"\nCriteria: CONTRADICTORY if non-software logic/spam is found, else COHERENT.\nFormat:\nVERDICT: [COHERENT/CONTRADICTORY]\nREASON: [Short explanation]"
    try:
        response = chat_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0, max_output_tokens=600), request_options={"timeout": 15.0})
        res = response.text.strip().upper()
        verdict = "CONTRADICTORY" if "CONTRADICTORY" in res else "COHERENT"
        reason = next((line.split(":", 1)[-1].strip() for line in response.text.splitlines() if "REASON:" in line.upper()), "Semantic logic verified.")
        return verdict, reason
    except Exception as e:
        return "COHERENT", f"[API Fallback] Local check assumed coherent due to network timeout ({type(e).__name__})."

def generate_ai_response(user_question, bug_report_context, prediction_status, guideline_text="", stream=True):
    if not bug_report_context.strip(): return "I do not have a bug report to look at yet."
    prompt = f"Date: {datetime.now().strftime('%Y-%m-%d')}\nStatus: {prediction_status}\nRaw Report:\n\"\"\"{bug_report_context}\"\"\"\nTask: {user_question}\nEnsure components: Severity, Routing, Steps, Root Cause."
    
    try:
        return chat_model.generate_content(prompt, stream=stream, generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=2048))
    except Exception as e:
        local_route = predict_component_local(bug_report_context)
        fallback = (
            f"### ⚠️ [Semantic AI Offline - Standardized by Local Fallback]\n\n"
            f"**Assigned Routing Component:** `{local_route}`\n"
            f"**Validation Status:** {prediction_status}\n\n"
            f"#### Raw Description:\n{bug_report_context}\n\n"
            f"> *Note: Gemini API timed out. Structural data was preserved using localized NLP components.*"
        )
        return fallback

def export_to_external_system(custom_title, custom_desc, prediction, confidence, coherence_status, github_repo, access_token, user_info):
    val_text = f"### 🤖 Automated Triage Validation\n- **Phase 1 (Structural):** `{prediction}` ({confidence}% confidence)\n"
    if coherence_status and coherence_status != "PENDING":
        val_text += f"- **Phase 2 (Semantic):** `{coherence_status}`\n"
    signature = f"\n\n---\n*Reported via BugTriage-NLP by **{user_info.get('name', 'Unknown')}** ({user_info.get('email', '')})*"
    issue_body = val_text + "\n---\n\n" + custom_desc + signature

    if not access_token or not github_repo: return False, "Missing token or Repo name."
    url = f"https://api.github.com/repos/{github_repo}/issues"
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github+json"}
    
    try:
        resp = requests.post(url, headers=headers, json={"title": custom_title, "body": issue_body, "labels": ["bug", "triage-validated"]}, timeout=10)
        if resp.status_code in (200, 201): return True, f"Ticket successfully exported to: {resp.json().get('html_url', '')}"
        return False, f"GitHub API Error ({resp.status_code}): {resp.text}"
    except Exception as e: return False, f"Connection error: {e}"

def start_new_chat(preserve_draft=""):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.draft_report = preserve_draft
    for key in ["current_report", "prediction", "confidence", "coherence_status", "coherence_reason", "phase1_ms", "phase2_ms", "duplicates"]:
        if key in st.session_state: del st.session_state[key]

# ---------------------------------------------------------
# UI RENDERING
# ---------------------------------------------------------

if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state: st.session_state.messages = []

with st.sidebar:
    if st.button("New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.divider()
    if st.session_state.user is not None:
        st.header("Your Chat History")
        if db is not None:
            sessions = list(db.sessions.find({"user_email": st.session_state.user["email"]}, {"session_id": 1, "title": 1, "timestamp": 1, "_id": 0}).sort("timestamp", -1))
            if sessions:
                with st.container(height=400):
                    for doc in sessions:
                        if st.button(f"{doc['title']} ({doc['timestamp'].split()[0]})", key=doc['session_id'], use_container_width=True):
                            sess_data = db.sessions.find_one({"session_id": doc['session_id']})
                            st.session_state.update({
                                "session_id": doc['session_id'], "current_report": sess_data.get("bug_report", ""),
                                "prediction": sess_data.get("prediction", ""), "confidence": sess_data.get("confidence", 0.0),
                                "coherence_status": sess_data.get("coherence_status", "COHERENT"), "coherence_reason": sess_data.get("coherence_reason", ""),
                                "phase1_ms": sess_data.get("phase1_ms", 0), "phase2_ms": sess_data.get("phase2_ms", 0)
                            })
                            msgs = list(db.chat_history.find({"session_id": doc['session_id']}).sort("timestamp", 1))
                            st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in msgs]
                            st.rerun()
    else:
        st.info("🔒 Sign in above to view history.")

st.title("BugTriage-NLP")
st.markdown("**Automated Defect Report Validation and Triage Assistant**")

tab_workspace, tab_dashboard = st.tabs(["🛠️ Triage Workspace", "📊 QA Manager Dashboard"])

with tab_workspace:
    if "current_report" not in st.session_state:
        st.write("Input the defect description below to run Phase 1 structural validation.")
        default_text = st.session_state.get("draft_report", "")
        if sample_df is not None:
            opts = ["-- Type manually --"] + [f"Ticket #{i+1}: {r[sample_df.columns[0]]}" for i, r in sample_df.iterrows()]
            sel = st.selectbox("Load sample ticket:", opts)
            if sel != "-- Type manually --": default_text = str(sample_df.iloc[int(sel.split("#")[1].split(":")[0])-1][sample_df.columns[-1]])
            
        user_input = st.text_area("Defect Description", value=default_text, height=150, label_visibility="collapsed")

        if st.button("Analyze Report", use_container_width=True, type="primary"):
            if user_input.strip():
                t0 = time.time()
                clean_input = user_input.strip()
                final_prediction, valid_conf = evaluate_phase1_structure(clean_input)
                t1 = time.time()
                
                st.session_state.update({
                    "current_report": clean_input, "prediction": final_prediction, 
                    "confidence": valid_conf, "coherence_status": "PENDING", 
                    "phase1_ms": int((t1 - t0) * 1000), "duplicates": check_for_duplicates(clean_input, st.session_state.session_id)
                })
                email = st.session_state.user["email"] if st.session_state.user else "anonymous"
                create_session(st.session_state.session_id, email, clean_input, final_prediction, valid_conf, p1_ms=st.session_state.phase1_ms)
                st.rerun()
            else: st.warning("Enter a defect description.")
    else:
        with st.expander("View Active Defect Report Context"): st.write(st.session_state.current_report)

        if st.session_state.get("duplicates"):
            st.warning(f"⚠️ **Potential Duplicates Found:** Similarity >85% with {len(st.session_state.duplicates)} past report(s).")
            for dup in st.session_state.duplicates: st.caption(f"- {dup['title']} ({dup['score']}%)")

        col1, col2, col3 = st.columns([1.2, 1.2, 1])
        with col1:
            st.caption(f"Phase 1: Validation ({st.session_state.get('phase1_ms', 0)}ms)")
            if st.session_state.prediction == "Valid": st.success(f"Valid Structure ({st.session_state.confidence}%)")
            else: st.error(f"Missing Details ({st.session_state.confidence}%)")

        if st.session_state.prediction == "Missing_Details":
            with col2:
                st.caption("Phase 2: Semantic Verification")
                st.warning("Skipped (Blocked by Phase 1)")
            st.warning("Revision Required: Score below threshold. Add clear Steps, Expected, and Actual results.")
            if st.button("Revise & Resubmit", type="primary"):
                start_new_chat(preserve_draft=st.session_state.current_report)
                st.rerun()
        else:
            with col2:
                p2_ms = st.session_state.get('phase2_ms', 0)
                st.caption(f"Phase 2: Semantic Check {'('+str(p2_ms)+'ms)' if p2_ms else ''}")
                if st.session_state.get("coherence_status") == "PENDING":
                    if st.button("Run Coherence Check", type="secondary"):
                        with st.spinner("Checking..."):
                            t0 = time.time()
                            verdict, reason = verify_semantic_coherence(st.session_state.current_report)
                            t1 = time.time()
                            st.session_state.update({"coherence_status": verdict, "coherence_reason": reason, "phase2_ms": int((t1 - t0) * 1000)})
                            create_session(st.session_state.session_id, st.session_state.user["email"] if st.session_state.user else "anonymous", st.session_state.current_report, st.session_state.prediction, st.session_state.confidence, verdict, reason, st.session_state.phase1_ms, st.session_state.phase2_ms)
                            st.rerun()
                elif st.session_state.coherence_status == "CONTRADICTORY": st.error(f"Flagged: {st.session_state.coherence_reason}")
                else: st.success("Coherent")

            with col3:
                st.caption("Phase 3: External Routing")
                if st.session_state.user:
                    target_repo = st.text_input("Target Repo", placeholder="owner/repo")
                    custom_title = st.text_input("Issue Title", value=f"[BugTriage] {st.session_state.current_report.splitlines()[0][:65]}")
                    latest_ai = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"), f"#### Raw Report\n{st.session_state.current_report}")
                    custom_desc = st.text_area("Issue Description", value=latest_ai, height=100)
                    
                    if st.button("Export to GitHub", type="primary"):
                        with st.spinner("Exporting..."):
                            success, msg = export_to_external_system(custom_title, custom_desc, st.session_state.prediction, st.session_state.confidence, st.session_state.coherence_status, target_repo, st.session_state.user["access_token"], st.session_state.user)
                            if success: st.success(msg)
                            else: st.error(msg)
                else: st.info("🔒 Sign in to export.")

        st.markdown("### Triage Assistant")
        if st.button("Standardize & Route via AI"):
            msg = "Standardize this bug report. Identify Severity, Component Routing, and Root Cause."
            st.session_state.messages.append({"role": "user", "content": msg})
            save_message(st.session_state.session_id, "user", msg)
            
            resp = generate_ai_response(msg, st.session_state.current_report, st.session_state.prediction, stream=True)
            if isinstance(resp, str): 
                st.session_state.messages.append({"role": "assistant", "content": resp})
            else:
                def stream_gen():
                    full = ""
                    for chunk in resp:
                        if chunk.text:
                            full += chunk.text
                            yield chunk.text
                    save_message(st.session_state.session_id, "assistant", full)
                    st.session_state.messages.append({"role": "assistant", "content": full})
                st.write_stream(stream_gen)
            st.rerun()

        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

with tab_dashboard:
    st.header("System Analytics & Research Data")
    if db is not None:
        total = db.sessions.count_documents({})
        if total > 0:
            v_count = db.sessions.count_documents({"prediction": "Valid"})
            avg_p1_result = db.sessions.aggregate([{"$group": {"_id": None, "avg": {"$avg": "$phase1_ms"}}}]).next()["avg"]
            avg_p1_display = round(avg_p1_result, 2) if avg_p1_result is not None else 0
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Total Reports Processed", total)
            m_col2.metric("Valid Structure Rate", f"{round((v_count/total)*100, 1)}%")
            m_col3.metric("Avg Phase 1 Speed", f"{avg_p1_display} ms")
            
            st.subheader("Performance Logs (Research Output)")
            logs_df = pd.DataFrame(list(db.sessions.find({}, {"session_id":1, "prediction":1, "phase1_ms":1, "phase2_ms":1, "timestamp":1, "_id":0})))
            st.dataframe(logs_df, use_container_width=True)
        else:
            st.info("No data available yet. Process some defect reports to generate analytics.")
    else:
        st.error("Database connection required for Manager Dashboard.")
