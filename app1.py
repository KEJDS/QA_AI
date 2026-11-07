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
from sklearn.feature_extraction.text import TfidfVectorizer
from authlib.integrations.requests_client import OAuth2Session
import io

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
# DOCUMENT PARSER
# ---------------------------------------------------------
def extract_text_from_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        elif uploaded_file.name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif uploaded_file.name.endswith('.txt'):
            return uploaded_file.getvalue().decode("utf-8")
        return ""
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

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
            st.session_state.prd_text = ""
            
            for key in ["current_report", "prediction", "confidence", "coherence_status", "coherence_reason", "draft_report", "phase1_ms", "phase2_ms", "duplicates", "coherence_confidence", "coherence_details"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.query_params.clear()
            st.rerun()

handle_oauth_login()

# ---------------------------------------------------------
# CORE SYSTEM INIT
# ---------------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_chat_model():
    if not GEMINI_API_KEY:
        return None
    try:
        available = [m.name for m in genai.list_models()
                     if "generateContent" in m.supported_generation_methods]
        preferred = ["models/gemini-2.5-flash", "models/gemini-2.0-flash",
                     "models/gemini-1.5-flash"]
        target = next((m for m in preferred if m in available),
                      available[0] if available else None)
        if not target:
            return None
        return genai.GenerativeModel(
            model_name=target,
            system_instruction="You are an expert Software QA triage specialist. Be concise."
        )
    except Exception:
        return None

@st.cache_resource
def load_local_ml_pipeline():
    return joblib.load(VEC_PATH), joblib.load(CLF_PATH)

@st.cache_data
def load_sample_tickets():
    return pd.read_csv(SAMPLE_CSV_PATH, encoding="utf-8") if os.path.exists(SAMPLE_CSV_PATH) else None

chat_model = load_chat_model()
gemini_available = chat_model is not None
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

# ---------------------------------------------------------
# PHASE 2: LOCAL COHERENCE CHECKER + OPTIONAL GEMINI
# ---------------------------------------------------------
SOFTWARE_TERMS = {
    "software", "system", "application", "app", "website", "webpage",
    "browser", "database", "sql", "mongo", "api", "endpoint", "server",
    "network", "request", "response", "login", "logout", "authentication",
    "authorization", "password", "account", "user", "button", "form",
    "interface", "ui", "ux", "frontend", "backend", "function", "feature",
    "code", "program", "script", "error", "exception", "crash", "bug",
    "defect", "issue", "page", "screen", "upload", "download", "file",
    "query", "table", "timeout", "loading", "submit", "save", "install",
    "version", "mobile", "android", "ios", "test", "testing", "performance",
    "configuration", "cache", "permission", "data"
}
EXPECTED_TERMS = ["expected", "should", "intended", "expected result", "expected behavior"]
ACTUAL_TERMS = ["actual", "observed", "currently", "instead", "actual result", "actual behavior"]
STEP_TERMS = ["steps to reproduce", "reproduction steps", "steps", "step 1", "reproduce", "when i"]

CONTRADICTION_RULES = [
    (r"\b(?:can|able to|successfully)\s+(?:log in|login)\b",
     r"\b(?:cannot|can't|unable to|fails to)\s+(?:log in|login)\b",
     "Conflicting login claims"),
    (r"\b(?:data|record|changes?)\s+(?:is|are)?\s*(?:saved|stored)\b",
     r"\b(?:data|record|changes?)\s+(?:is|are)?\s*(?:not saved|not stored|unsaved)\b",
     "Conflicting data-persistence claims"),
    (r"\b(?:button|control)\s+(?:is|appears)?\s*enabled\b",
     r"\b(?:button|control)\s+(?:is|appears)?\s*disabled\b",
     "Conflicting button-state claims"),
    (r"\b(?:page|application|system)\s+(?:loads|opens|works|responds)\b",
     r"\b(?:page|application|system)\s+(?:does not load|doesn't load|fails to open|is not responding)\b",
     "Conflicting system-state claims"),
    (r"\b(?:user|admin)\s+(?:has|have)\s+access\b",
     r"\b(?:user|admin)\s+(?:does not|doesn't|has no)\s+access\b",
     "Conflicting access claims"),
]

def _has_any(text, phrases):
    return any(p in text for p in phrases)

def calculate_relevance_score(text):
    lower = text.lower()
    matched = sorted({
        term for term in SOFTWARE_TERMS
        if re.search(r"\b" + re.escape(term) + r"\b", lower)
    })
    patterns = [
        r"\bhttps?://", r"\b(?:500|404|401|403|502|503)\b",
        r"\b(?:exception|traceback|stack trace|failed|failure)\b",
        r"\b(?:click|tap|select|enter|navigate|open|submit)\b",
    ]
    pattern_hits = sum(bool(re.search(p, lower)) for p in patterns)
    score = min(75, len(matched) * 7.5)
    score += min(15, pattern_hits * 5)
    if _has_any(lower, EXPECTED_TERMS + ACTUAL_TERMS + STEP_TERMS):
        score += 10
    return round(min(100, score), 1), matched

def calculate_sentence_similarity(text):
    sentences = [
        re.sub(r"\s+", " ", x).strip()
        for x in re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(x.strip().split()) >= 3
    ]
    if len(sentences) < 2:
        return None
    try:
        local_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = local_vec.fit_transform(sentences)
        sim = cosine_similarity(matrix)
        vals = [float(sim[i, j]) for i in range(len(sentences))
                for j in range(i + 1, len(sentences))]
        return round(100 * sum(vals) / len(vals), 1) if vals else None
    except ValueError:
        return None

def detect_contradictions(text):
    lower = text.lower()
    findings = []
    for positive, negative, label in CONTRADICTION_RULES:
        if re.search(positive, lower) and re.search(negative, lower):
            findings.append(label)

    expected = re.search(
        r"(?:expected(?: result| behavior)?|should be)\s*[:\-]?\s*(.{5,180})",
        lower
    )
    actual = re.search(
        r"(?:actual(?: result| behavior)?|observed)\s*[:\-]?\s*(.{5,180})",
        lower
    )
    if expected and actual:
        e, a = expected.group(1), actual.group(1)
        opposites = [
            ("visible", "not visible"), ("displayed", "not displayed"),
            ("enabled", "disabled"), ("successful", "failed"),
            ("accepted", "rejected"), ("present", "missing"),
            ("connected", "disconnected"), ("saved", "not saved"),
        ]
        for positive, negative in opposites:
            if ((positive in e and negative in a) or
                (negative in e and positive in a)):
                findings.append(f"Expected and actual conflict: {positive}/{negative}")
    return list(dict.fromkeys(findings))

def verify_semantic_coherence(bug_report: str, use_gemini=False):
    text = (bug_report or "").strip()
    lower = text.lower()
    words = re.findall(r"\b\w+\b", text)
    
    # Initialize scope variables to prevent UnboundLocalError
    confidence = 50.0
    metrics = {
        "relevance": 0.0, "similarity": None,
        "contradictions": [], "matched_terms": [],
        "word_count": len(words), "has_steps": False,
        "has_expected": False, "has_actual": False,
        "absurd_hits": 0
    }

    if not text:
        return "UNVERIFIED", "No report provided.", confidence, metrics

    relevance, matched = calculate_relevance_score(text)
    similarity = calculate_sentence_similarity(text)
    contradictions = detect_contradictions(text)
    
    absurd_patterns = [
        r"\b(?:toaster|bread|pumpernickel|celery|potato|soup|seagulls|purple|caramelize|origami|swan|hieroglyphics)\b",
        r"\b(?:smells like|tastes like|singing|humming|stare intensely|facing north|whisper a polite apology)\b",
        r"\b(?:sentient|unionized|smooth jazz|victorian ghost|sun blowing up|line of time)\b",
        r"\b(?:windows 1998 se|soup edition)\b"
    ]
    absurd_hits = [p for p in absurd_patterns if re.search(p, lower)]
    
    has_expected = _has_any(lower, EXPECTED_TERMS)
    has_actual = _has_any(lower, ACTUAL_TERMS)
    has_steps = _has_any(lower, STEP_TERMS)
    section_count = sum([has_expected, has_actual, has_steps])

    confidence = 25 + min(30, relevance * .30) + section_count * 10
    if similarity is not None:
        confidence += min(15, similarity * .15)
    if len(words) >= 30:
        confidence += 5
    if contradictions or absurd_hits:
        confidence = max(confidence, 75)
    confidence = round(min(95, confidence), 1)

    metrics.update({
        "relevance": relevance, "similarity": similarity,
        "contradictions": contradictions, "matched_terms": matched,
        "has_steps": has_steps, "has_expected": has_expected, 
        "has_actual": has_actual, "absurd_hits": len(absurd_hits)
    })

    if use_gemini and chat_model is not None:
        prompt = (
            "You are a zero-tolerance software quality assurance gatekeeper.\n\n"
            f"Analyze this defect report word-for-word:\n\"\"\"{text}\"\"\"\n\n"
            "EVALUATION INSTRUCTIONS:\n"
            "1. Scan the text for ANY slang, pop-culture references, jokes, song lyrics, or physically impossible events (e.g., database caramelizing, browser made of bread), no matter how much valid technical code is present.\n"
            "2. If you find EVEN ONE such phrase, your verdict MUST be CONTRADICTORY.\n"
            "3. Only output COHERENT if 100% of the text describes a legitimate software problem or configuration.\n\n"
            "Respond EXACTLY in this format:\n"
            "VERDICT: [COHERENT or CONTRADICTORY]\n"
            "REASON: [If contradictory, quote the exact bad phrase. If coherent, write 'Semantic logic verified']"
        )
        try:
            response = chat_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0, max_output_tokens=300
                ),
                request_options={"timeout": 15.0},
            )
            res = response.text.upper()
            verdict = "CONTRADICTORY" if "CONTRADICTORY" in res else "COHERENT"
            
            reason = "Semantic logic verified."
            for line in response.text.splitlines():
                if "REASON:" in line.upper():
                    reason = line.split(":", 1)[-1].replace("**", "").strip()
                    break
            return verdict, reason, max(confidence, 85), metrics
        except Exception as exc:
            metrics["gemini_error"] = type(exc).__name__

    if absurd_hits:
        return "CONTRADICTORY", "Local anomaly rule: Detected surreal, satirical, or physically impossible claims (e.g., sentient hardware or absurd environmental steps).", confidence, metrics

    if contradictions:
        return "CONTRADICTORY", "Local contradiction rule(s): " + "; ".join(contradictions), confidence, metrics
    if len(words) < 15:
        return "UNVERIFIED", "Report is too short for reliable checking. Manual review required.", min(confidence, 45), metrics
    if relevance < 20:
        return "UNVERIFIED", "Low software relevance score. Manual review required.", min(confidence, 50), metrics
    if section_count < 2:
        return "UNVERIFIED", "Insufficient expected/actual/steps evidence. Manual review required.", min(confidence, 60), metrics

    return "COHERENT", "Local checks passed. This does not guarantee semantic correctness.", confidence, metrics

def generate_ai_response(user_question, bug_report_context, prediction_status, guideline_text="", stream=True):
    if not bug_report_context.strip(): return "I do not have a bug report to look at yet."
    prompt = f"Date: {datetime.now().strftime('%Y-%m-%d')}\nStatus: {prediction_status}\nRaw Report:\n\"\"\"{bug_report_context}\"\"\"\nTask: {user_question}\nEnsure components: Severity, Routing, Steps, Root Cause."
    
    try:
        if chat_model is None:
            raise RuntimeError("Gemini unavailable or API key not configured.")
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
    for key in ["current_report", "prediction", "confidence", "coherence_status", "coherence_reason", "phase1_ms", "phase2_ms", "duplicates", "coherence_confidence", "coherence_details"]:
        if key in st.session_state: del st.session_state[key]

# ---------------------------------------------------------
# UI RENDERING
# ---------------------------------------------------------

if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state: st.session_state.messages = []
if "prd_text" not in st.session_state: st.session_state.prd_text = ""

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
st.caption("Gemini available" if gemini_available else "Offline-capable mode · Gemini is optional")

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
            
            if st.button("🔄 Re-run Phase 1", use_container_width=True):
                with st.spinner("Re-evaluating structural integrity..."):
                    t0 = time.time()
                    final_prediction, valid_conf = evaluate_phase1_structure(st.session_state.current_report)
                    t1 = time.time()
                    st.session_state.update({
                        "prediction": final_prediction,
                        "confidence": valid_conf,
                        "phase1_ms": int((t1 - t0) * 1000)
                    })
                    email = st.session_state.user["email"] if st.session_state.user else "anonymous"
                    create_session(st.session_state.session_id, email, st.session_state.current_report, final_prediction, valid_conf, p1_ms=st.session_state.phase1_ms, p2_ms=st.session_state.get("phase2_ms", 0))
                    st.success("Phase 1 re-evaluated successfully!")
                    st.rerun()

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
                use_gemini_phase2 = st.checkbox(
                    "Use Gemini for enhanced semantic review",
                    value=False,
                    disabled=not gemini_available,
                    help="Local checks always run. Gemini is optional."
                )
                if not gemini_available:
                    st.caption("Gemini unavailable — local checker is ready.")

                btn_label = "Run Coherence Check" if st.session_state.get("coherence_status", "PENDING") == "PENDING" else "🔄 Re-run Coherence Check"
                
                if st.button(btn_label, type="secondary", use_container_width=True):
                    with st.spinner("Running coherence checks..."):
                        t0 = time.time()
                        verdict, reason, p2_confidence, p2_metrics = verify_semantic_coherence(
                            st.session_state.current_report,
                            use_gemini=use_gemini_phase2
                        )
                        t1 = time.time()
                        st.session_state.update({
                            "coherence_status": verdict,
                            "coherence_reason": reason,
                            "coherence_confidence": p2_confidence,
                            "coherence_details": p2_metrics,
                            "phase2_ms": int((t1 - t0) * 1000)
                        })
                        create_session(
                            st.session_state.session_id,
                            st.session_state.user["email"] if st.session_state.user else "anonymous",
                            st.session_state.current_report,
                            st.session_state.prediction,
                            st.session_state.confidence,
                            verdict, reason,
                            st.session_state.phase1_ms,
                            st.session_state.phase2_ms
                        )
                        st.rerun()

                if st.session_state.get("coherence_status") == "CONTRADICTORY":
                    st.error(f"Flagged: {st.session_state.coherence_reason}")
                    st.metric("Coherence confidence", f"{st.session_state.get('coherence_confidence', 0)}%")
                elif st.session_state.get("coherence_status") == "UNVERIFIED":
                    st.warning(f"Manual review required: {st.session_state.coherence_reason}")
                    st.metric("Coherence confidence", f"{st.session_state.get('coherence_confidence', 0)}%")
                elif st.session_state.get("coherence_status") == "PENDING":
                    st.info("Awaiting Coherence Verification")
                else:
                    st.success(st.session_state.coherence_reason)
                    st.metric("Coherence confidence", f"{st.session_state.get('coherence_confidence', 0)}%")

                if st.session_state.get("coherence_details"):
                    with st.expander("View local coherence metrics"):
                        details = st.session_state.coherence_details
                        st.metric("Software relevance", f"{details.get('relevance', 0)}%")
                        sim = details.get("similarity")
                        st.metric("Sentence similarity", "Not available" if sim is None else f"{sim}%")
                        st.caption("Matched terms: " + (", ".join(details.get("matched_terms", [])) or "None"))

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

        st.markdown("### 🤖 Triage Assistant & PRD Verification")
        col_ai1, col_ai2 = st.columns(2)
        
        with col_ai1:
            st.markdown("#### Standard Route")
            if st.button("Standardize & Route via AI", use_container_width=True):
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

        with col_ai2:
            st.markdown("#### Requirement Cross-Check")
            
            if not st.session_state.get("prd_text"):
                st.info("⚠️ You must upload a reference document (PRD) to verify the expected behavior.")
                uploaded_prd = st.file_uploader("Upload PRD (PDF, Word, TXT)", type=['pdf', 'docx', 'txt'], label_visibility="collapsed")
                if uploaded_prd:
                    st.session_state.prd_text = extract_text_from_file(uploaded_prd)
                    if st.session_state.prd_text:
                        st.success("✅ Requirements loaded! You can now verify.")
                        st.rerun()
            else:
                st.success("✅ Reference Document is actively loaded.")
                if gemini_available:
                    if st.button("Verify against Reference", use_container_width=True, type="primary"):
                        msg = "Verify this bug report's 'Expected Behavior' against the uploaded Product Requirements Document (PRD)."
                        st.session_state.messages.append({"role": "user", "content": msg})
                        
                        prompt = (
                            "You are a strict QA Lead. Compare the following Bug Report against the provided Product Requirements Document (PRD).\n\n"
                            "INSTRUCTIONS:\n"
                            "1. Verify if the 'Expected Behavior' in the bug report aligns with the PRD.\n"
                            "2. State whether the bug is valid (a real deviation from requirements) or invalid (works as designed according to the PRD).\n"
                            "3. Quote the specific section of the PRD that proves your point.\n\n"
                            f"--- PRD EXTRACT ---\n{st.session_state.prd_text[:12000]}\n\n"
                            f"--- BUG REPORT ---\n{st.session_state.current_report}"
                        )
                        
                        try:
                            with st.spinner("Cross-referencing business rules..."):
                                resp = chat_model.generate_content(prompt)
                                st.session_state.messages.append({"role": "assistant", "content": resp.text})
                                save_message(st.session_state.session_id, "assistant", resp.text)
                        except Exception as e:
                            st.error(f"Error checking requirements: {e}")
                        st.rerun()
                
                if st.button("Clear Reference Document", use_container_width=True):
                    st.session_state.prd_text = ""
                    st.rerun()

        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

with tab_dashboard:
    st.header("System Analytics & Research Data")
    
    if st.session_state.user is None:
        st.warning("🔒 Please sign in with GitHub to view your analytics and performance logs.")
    elif db is not None:
        user_email = st.session_state.user["email"]
        total = db.sessions.count_documents({"user_email": user_email})
        
        if total > 0:
            v_count = db.sessions.count_documents({"user_email": user_email, "prediction": "Valid"})
            agg_pipeline = [
                {"$match": {"user_email": user_email}},
                {"$group": {"_id": None, "avg": {"$avg": "$phase1_ms"}}}
            ]
            avg_p1_cursor = list(db.sessions.aggregate(agg_pipeline))
            avg_p1_result = avg_p1_cursor[0]["avg"] if avg_p1_cursor else None
            avg_p1_display = round(avg_p1_result, 2) if avg_p1_result is not None else 0
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Your Reports Processed", total)
            m_col2.metric("Valid Structure Rate", f"{round((v_count/total)*100, 1)}%")
            m_col3.metric("Avg Phase 1 Speed", f"{avg_p1_display} ms")
            
            st.subheader("Performance Logs (Research Output)")
            logs_df = pd.DataFrame(list(db.sessions.find(
                {"user_email": user_email}, 
                {"session_id":1, "prediction":1, "phase1_ms":1, "phase2_ms":1, "timestamp":1, "_id":0}
            )))
            st.dataframe(logs_df, use_container_width=True)
        else:
            st.info("No data available yet. Process some defect reports to generate analytics.")
    else:
        st.error("Database connection required for the Dashboard.")
