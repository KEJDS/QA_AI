import streamlit as st
import joblib
import google.generativeai as genai

# API and modeling 

# Your Google API key
# Safely pull the API key from Streamlit Cloud's secret manager
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_resource
def load_chat_model():
    """
    Directly loads the specific model version recommended by the Google API 
    to bypass any deprecation or 404 errors.
    """
    return genai.GenerativeModel('models/gemini-3.6-flash')

# Initialize the cloud model
chat_model = load_chat_model()

# Load your local ML classification model for the initial validation gate
model = joblib.load("bug_model.pkl")

#help instructions

from datetime import datetime

def generate_ai_response(user_question, bug_report_context, prediction_status, stream=False):
    """
    100% flexible AI processing. No hardcoded conversational rules.
    Injects real-time system context so the LLM can answer naturally.
    """
    if not bug_report_context or bug_report_context.strip() == "":
        return "I don't have a bug report to look at yet! Please analyze one first."
        
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Notice we give the AI the 'System Context' so it can figure out the answer itself
    prompt = f"""
    You are an expert Software Quality Assurance Engineer assisting a software developer.
    System Context: Today's date is {current_date}.
    
    You are reviewing the following bug report:
    "{bug_report_context}"
    
    The initial automated validation system flagged this report as: {prediction_status}.
    (Note: "Valid" means it has good structure. "Missing_Details" means it lacks steps or expected outcomes).
    
    The user is asking you this question: "{user_question}"
    
    Answer the user's question directly, conversationally, and practically. 
    Analyze the report to suggest potential root causes—such as environment configuration, frontend UI logic, backend routing, or database execution errors—based strictly on the context provided.
    If they ask an unrelated question, answer it briefly if you have the context, but gently steer them back to discussing the bug report.
    """
    
    try:
        response = chat_model.generate_content(prompt, stream=stream)
        return response if stream else response.text
    except Exception as e:
        return f"I encountered an error connecting to the cloud AI: {e}"

#app logic and streamlit UI

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="BugTriage-NLP", layout="centered")

st.title("BugTriage-NLP")
st.subheader("Automated Defect Report Validation & Cloud AI Assistant")
st.write("Drop your bug report below. The local ML model will validate it, and the cloud AI will help you refine it.")

user_input = st.text_area("Defect Description:", height=150, placeholder="Type or paste your bug report here...")

if st.button("Analyze Report"):
    if user_input.strip() == "":
        st.warning("Please enter a defect description before analyzing.")
    else:
        st.session_state.current_report = user_input
        st.session_state.messages = [] 
        
        # Hybrid validation check
        text_lower = user_input.lower()
        has_structure = ("steps" in text_lower or "reproduce" in text_lower) and ("expected" in text_lower or "actual" in text_lower)
        
        prediction = model.predict([user_input])[0]
        
        if has_structure or prediction == "Valid":
            st.session_state.prediction = "Valid"
            st.success("Looks solid! This report has enough technical depth for triage.")
        else:
            st.session_state.prediction = "Missing_Details"
            st.error("This report is a bit thin on details. It's missing clear steps or expected outcomes.")

if "current_report" in st.session_state:
    st.markdown("---")
    st.markdown("### 💬 Triage Assistant")
    
    # Quick action buttons for the LLM
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Analyze this report deeply"):
            st.session_state.messages.append({"role": "user", "content": "Analyze this report deeply and tell me what the core issue likely is."})
            with st.spinner("Analyzing..."):
                reply = generate_ai_response("Analyze this report deeply and tell me what the core issue likely is.", st.session_state.current_report, st.session_state.prediction, stream=False)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

    with col2:
        if st.button("📝 Rewrite professionally"):
            st.session_state.messages.append({"role": "user", "content": "Rewrite this bug report so it is perfectly formatted for a developer."})
            with st.spinner("Rewriting..."):
                reply = generate_ai_response("Rewrite this bug report so it is perfectly formatted for a developer with sections for Environment, Steps, Expected, and Actual results.", st.session_state.current_report, st.session_state.prediction, stream=False)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user text input
    if prompt_input := st.chat_input("Ask me anything about this report..."):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            response_stream = generate_ai_response(
                user_question=prompt_input, 
                bug_report_context=st.session_state.current_report,
                prediction_status=st.session_state.prediction,
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