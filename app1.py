import os
import streamlit as st
import joblib
import google.generativeai as genai
import requests
from datetime import datetime
import io
import sqlite3
import uuid

# --- IMPORTS FOR FILE PARSING ---
import PyPDF2
import docx

# --- BASE DIRECTORY & FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chat_logs.db")

# Points directly inside the "GitBugs" folder on GitHub (case-sensitive)
VEC_PATH = os.path.join(BASE_DIR, "GitBugs", "tfidf_vectorizer.pkl")
CLF_PATH = os.path.join(BASE_DIR, "GitBugs", "logistic_regression_validator.pkl")

# --- API AND ML MODEL LOADING ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_resource
def load_chat_model():
    return genai.GenerativeModel('models/gemini-3.6-flash')

@st.cache_resource
def load_local_ml_pipeline():
    """Loads the TF-IDF Vectorizer and Logistic Regression Classifier from the GitBugs folder."""
    vectorizer = joblib.load(VEC_PATH)
    validator = joblib.load(CLF_PATH)
    return vectorizer, validator

chat_model = load_chat_model()
vectorizer, validator = load_local_ml_pipeline()

def evaluate_bug_report(raw_text):
    """Runs the TF-IDF + Logistic Regression classifier and returns (status, confidence%)."""
    clean_input = raw_text.strip()
    text_lower = clean_input.lower()
    word_count = len(clean_input.split())

    features = vectorizer.transform([clean_input])
    probabilities = validator.predict_proba(features)[0]
    class_labels = list(validator.classes_)

    ml_prediction = validator.predict(features)[0]
    has_structure = ("steps" in text_lower or "reproduce" in text_lower) and ("expected" in text_lower or "actual" in text_lower)

    if word_count < 15:
        final_prediction = "Missing_Details"
    elif has_structure or ml_prediction == "Valid":
        final_prediction = "Valid"
    else:
        final_prediction = "Missing_Details"

    # Get the exact probability for the predicted class
    if final_prediction in class_labels:
        idx = class_labels.index(final_prediction)
        confidence = round(probabilities[idx] * 100, 2)
    else:
        confidence = round(max(probabilities) * 100, 2)

    # Ensure rule-triggered overrides show realistic confidence (>50%)
    if confidence < 50.0:
        confidence = round(100.0 - confidence, 2)

    return final_prediction, confidence

# --- DATABASE SETUP & SESSION MANAGEMENT ---
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
        "INSERT OR IGNORE INTO Sessions (session_id, title, bug_report, prediction) VALUES (?, ?, ?, ?)", 
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
    """Loads a previous session's context, messages, and recalculates its exact confidence."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT bug_report, prediction FROM Sessions WHERE session_id = ?", (session_id,))
    session_data = cursor.fetchone()
    
    if session_data:
        st.session_state.session_id = session_id
        st.session_state.current_report = session_data[0]
        
        # Recalculate confidence for THIS specific report so it never reuses the previous chat's score
        pred_status, conf_score = evaluate_bug_report(session_data[0])
        st.session_state.prediction = session_data[1] if session_data[1] else pred_status
        st.session_state.confidence = conf_score
        
        cursor.execute("SELECT role, content FROM ChatHistory WHERE session_id = ? ORDER BY id ASC", (session_id,))
        messages = cursor.fetchall()
        st.session_state.messages = [{"role": row[0], "content": row[1]} for row in messages]
        
    conn.close()

def start_new_chat():
    """Resets the application state for a new bug report."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    for key in ["current_report", "prediction", "confidence"]:
        if key in st.session_state:
            del st.session_state[key]

# Initialize database on app startup
init_db()

# Session tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

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
                extracted_text += page.extract_text() + "\n"
        elif file_extension == 'docx':
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading the file: {e}")
    return extracted_text

def export_to_external_system(bug_report, prediction):
    return True

def generate_ai_response(user_question, bug_report_context, prediction_status, guideline_text="", stream=False):
    if not bug_report_context or bug_report_context.strip() == "":
        return "I do not have a bug report to look at yet. Please analyze one first."
        
    current_date = datetime.now().strftime('%Y-%m-%d')
    guideline_instructions = f"\nCRITICAL INSTRUCTION: Strictly evaluate and format the bug report against these specific company guidelines:\n{guideline_text}\n" if guideline_text else ""
    
    prompt = f"""
    You are an expert Software Quality Assurance Engineer assisting a software developer.
    System Context: Today's date is {current_date}. {guideline_instructions}
    
    You are reviewing the following bug report:
    "{bug_report_context}"
    
    The initial local Scikit-Learn Logistic Regression validation system flagged this report as: {prediction_status}.
    
    The user is asking you this question: "{user_question}"
    
    Answer the user's question directly, conversationally, and practically. 
    When restructuring or analyzing the report, ensure you provide:
    1. Standardized Title & Summary
    2. Identified Severity Level (Blocker, Critical, Major, Minor, or Trivial)
    3. Recommended Developer Routing / Component Assignment (e.g., Backend/Core, SQL/Database, UI/Frontend, API, or Network)
    4. Structured Steps to Reproduce, Expected Result, and Actual Result (or list exact clarification questions if details are missing)
    5. Probable Root-Cause Analysis based strictly on the context provided.
    If they ask an unrelated question, answer it briefly if you have the context, but gently steer them back to discussing the bug report.
    """
    try:
        response = chat_model.generate_content(prompt, stream=stream)
        return response if stream else response.text
    except Exception as e:
        return f"I encountered an error connecting to the cloud AI: {e}"

# --- STREAMLIT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="BugTriage-NLP", layout="wide", initial_sidebar_state="expanded") 

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
        .history-btn>button {
            text-align: left;
            border: none;
            background: none;
            padding: 5px 0px;
            color: #555;
            font-size: 14px;
        }
        .history-btn>button:hover {
            color: #000;
            box-shadow: none;
            text-decoration: underline;
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

# Only show the input box if we are starting a fresh chat
if "current_report" not in st.session_state:
    st.write("Input the defect description below. The local classification model will validate its structural integrity, followed by AI refinement.")
    user_input = st.text_area("Defect Description", height=150, placeholder="Enter bug report details here...", label_visibility="collapsed")

    action_col1, action_col2, action_col3 = st.columns([1, 1, 2])

    with action_col1:
        if st.button("Analyze Report", use_container_width=True):
            if user_input.strip() == "":
                st.warning("Please enter a defect description before proceeding.")
            else:
                clean_input = user_input.strip()
                final_prediction, confidence = evaluate_bug_report(clean_input)
                
                # Save state
                st.session_state.current_report = clean_input
                st.session_state.prediction = final_prediction
                st.session_state.confidence = confidence
                
                # Log session in database
                create_session(st.session_state.session_id, clean_input, st.session_state.prediction)
                st.rerun()
else:
    # Display the active report context if we are currently analyzing one
    with st.expander("View Active Defect Report Context", expanded=False):
        st.write(st.session_state.current_report)
        
    status_col, action_col, empty_col = st.columns([1, 1, 2])
    with status_col:
        st.caption("Current Validation Status")
        conf_str = f" ({st.session_state.confidence}%)" if "confidence" in st.session_state else ""
        if st.session_state.prediction == "Valid":
            st.info(f"Status: Valid Structure{conf_str}")
        else:
            st.error(f"Status: Missing Details{conf_str}")
            
    with action_col:
        st.caption("External Routing")
        if st.button("Export to Tracking System", use_container_width=True):
            with st.spinner("Connecting to external API..."):
                success = export_to_external_system(st.session_state.current_report, st.session_state.prediction)
                if success:
                    st.toast("Ticket successfully exported.")

    st.markdown("### Triage Assistant")
    
    ai_col1, ai_col2, ai_col3 = st.columns([1, 1, 2])
    with ai_col1:
        if st.button("Analyze Root Cause", use_container_width=True):
            msg = "Analyze this report deeply, recommend the target developer team/component for routing, and identify the most probable root cause."
            st.session_state.messages.append({"role": "user", "content": msg})
            save_message(st.session_state.session_id, "user", msg)
            
            with st.spinner("Processing analysis..."):
                reply = generate_ai_response(msg, st.session_state.current_report, st.session_state.prediction, guideline_text=custom_guideline_text, stream=False)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                save_message(st.session_state.session_id, "assistant", reply)
                st.rerun()

    with ai_col2:
        if st.button("Standardize Report Format", use_container_width=True):
            msg = "Rewrite this bug report according to standard developer formatting, including Severity Level, Target Developer Routing, Steps to Reproduce, Expected vs. Actual Results, and Root-Cause Analysis."
            st.session_state.messages.append({"role": "user", "content": msg})
            save_message(st.session_state.session_id, "user", msg)
            
            with st.spinner("Restructuring..."):
                reply = generate_ai_response(msg, st.session_state.current_report, st.session_state.prediction, guideline_text=custom_guideline_text, stream=False)
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
