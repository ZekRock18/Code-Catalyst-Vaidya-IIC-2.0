import streamlit as st
import pypdf
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from .env file
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)

# Custom CSS for professional UI
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.stApp {
    background-color: #f4f6f9;
}
.main-container {
    background-color: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    padding: 2rem;
    margin: 1rem;
}
.stFileUploader > div > div > div > div {
    background-color: #e9ecef;
    border-radius: 10px;
}
.stSpinner > div {
    border-color: #4CAF50 transparent #4CAF50 transparent;
}
h1 {
    color: #2c3e50;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file(s)"""
    texts = []
    pdf_reader = pypdf.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        texts.append(page.extract_text())
    return " ".join(texts)

def generate_health_recommendations(extracted_text):
    """Generate health recommendations using Groq API"""
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional medical analysis AI. Provide comprehensive, evidence-based health recommendations with clear, actionable insights."
                },
                {
                    "role": "user", 
                    "content": f"Analyze this report as if the user does not have any idea about the report and the person is very least knowledgeable in medical field. so analyse the report and tell the person in his language what problem is he actually facing and what precautions and medications also his daily life style should he follow based on the report.\n\n{extracted_text}"
                }
            ],
            model="llama3-70b-8192"
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Analysis error: {str(e)}"

def app():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.title("🩺 AI Medical Report Analyzer")
    
    st.markdown("""
    ### Upload Your Medical Documents
    *Securely analyze lab reports and medical records with advanced AI*
    """)
    
    uploaded_files = st.file_uploader(
        "Choose PDF Files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload medical reports for comprehensive AI analysis"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
                recommendations = generate_health_recommendations(extracted_text)
                
                st.success(f"Analysis Complete: {uploaded_file.name}")
                
                # Expandable section for recommendations
                with st.expander("View Detailed Recommendations"):
                    st.write(recommendations)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()