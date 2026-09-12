import streamlit as st
import joblib
import google.generativeai as genai
import requests # Added for external system API connections
from datetime import datetime

# API and modeling 
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_resource
def load_chat_model():
    return genai.GenerativeModel('models/gemini-3.6-flash')

chat_model = load_chat_model()
model = joblib.load("bug_model.pkl")

# --- NEW FEATURE: External System Connector ---
def export_to_external_system(bug_report, prediction):
    """
    Placeholder function to send data to an external system like Jira or a SQL database.
    You would replace the URL and payload with the specific system's API requirements.
    """
    api_url = "https://api.your-external-system.com/v1/tickets"
    payload = {
        "title": f"New Defect - {prediction}",
        "description": bug_report,
        "priority": "High" if prediction == "Missing_Details" else "Normal"
    }
    
    # In a real scenario, you would use: response = requests.post(api_url, json=payload)
    # For now, we simulate a successful connection:
    return True

# --- UPDATED AI FUNCTION: Now accepts guidelines ---
def generate_ai_response(user_question, bug_report_context, prediction_status, guideline_text="", stream=False):
    if not bug_report_context or bug_report_context.strip() == "":
        return "I don't have a bug report to look at yet! Please analyze one first."
        
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Inject the uploaded guidelines if they exist
    guideline_instructions = ""
    if guideline_text:
        guideline_instructions = f"\nCRITICAL INSTRUCTION: Strictly evaluate the bug report against these specific company guidelines:\n{guideline_text}\n"
    
    prompt = f"""
    You are an expert Software Quality Assurance Engineer assisting a software developer.
    System Context: Today's date is {current_date}. {guideline_instructions}
    
    You are reviewing the following bug report:
    "{bug_report_context}"
    
    The initial automated validation system flagged this report as: {prediction_status}.
    
    The user is asking you this question: "{user_question}"
    
    Answer the user's question directly, conversationally, and practically. 
    Analyze the report to suggest potential root causes based strictly on the context provided.
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

st.set_page_config(page_title="BugTriage-NLP", layout="centered")

# Sidebar for File Uploads
with st.sidebar:
    st.header("⚙️ QA Settings")
    st.write("Upload specific QA formatting guidelines for the AI to follow.")
    uploaded_guideline = st.file_uploader("Upload Guidelines (.txt)", type=["txt"])
    
    custom_guideline_text = ""
    if uploaded_guideline is not None:
        # Read the text from the uploaded file
        custom_guideline_text = str(uploaded_guideline.read(), "utf-8")
        st.success("Guidelines loaded successfully!")

st.title("BugTriage-NLP")
st.subheader("Automated Defect Report Validation & Cloud AI Assistant")
st.write("Drop your bug report below. The local ML model will validate it, and the cloud AI will help you refine it.")

user_input = st.text_area("Defect Description:", height=150, placeholder="Type or paste your bug report here...")

# Quick action buttons in a row
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("Analyze Report"):
        if user_input.strip() == "":
            st.warning("Please enter a defect description before analyzing.")
        else:
            st.session_state.current_report = user_input
            st.session_state.messages = [] 
            
            text_lower = user_input.lower()
            has_structure = ("steps" in text_lower or "reproduce" in text_lower) and ("expected" in text_lower or "actual" in text_lower)
            prediction = model.predict([user_input])[0]
            
            if has_structure or prediction == "Valid":
                st.session_state.prediction = "Valid"
                st.success("Looks solid! This report has enough technical depth for triage.")
            else:
                st.session_state.prediction = "Missing_Details"
                st.error("This report is a bit thin on details. It's missing clear steps or expected outcomes.")

with btn_col2:
    if "current_report" in st.session_state:
        if st.button("📤 Export to External System"):
            with st.spinner("Connecting to API..."):
                # Call our external connection function
                success = export_to_external_system(st.session_state.current_report, st.session_state.prediction)
                if success:
                    st.success("Successfully pushed ticket to the external tracking system!")

if "current_report" in st.session_state:
    st.markdown("---")
    st.markdown("### 💬 Triage Assistant")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Analyze this report deeply"):
            st.session_state.messages.append({"role": "user", "content": "Analyze this report deeply and tell me what the core issue likely is."})
            with st.spinner("Analyzing..."):
                reply = generate_ai_response(
                    "Analyze this report deeply and tell me what the core issue likely is.", 
                    st.session_state.current_report, 
                    st.session_state.prediction,
                    guideline_text=custom_guideline_text, # Passing the uploaded guidelines
                    stream=False
                )
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

    with col2:
        if st.button("📝 Rewrite professionally"):
            st.session_state.messages.append({"role": "user", "content": "Rewrite this bug report so it is perfectly formatted for a developer."})
            with st.spinner("Rewriting..."):
                reply = generate_ai_response(
                    "Rewrite this bug report so it is perfectly formatted for a developer.", 
                    st.session_state.current_report, 
                    st.session_state.prediction,
                    guideline_text=custom_guideline_text, # Passing the uploaded guidelines
                    stream=False
                )
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_input := st.chat_input("Ask me anything about this report..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            response_stream = generate_ai_response(
                user_question=prompt_input, 
                bug_report_context=st.session_state.current_report,
                prediction_status=st.session_state.prediction,
                guideline_text=custom_guideline_text, # Passing the uploaded guidelines
                stream=True
            )
            
            if isinstance(response_stream, str):
                st.markdown(response_stream)
                st.session_state.messages.append({"role": "assistant", "content": response_stream})
            else:
                def stream_text():
                    for chunk in response_stream:
                        yield chunk.text
                
                full_reply = st.write_stream(stream_text)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})