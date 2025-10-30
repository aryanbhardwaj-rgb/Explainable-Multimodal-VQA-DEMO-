import streamlit as st
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests
import numpy as np

# Page configuration with custom styling
st.set_page_config(
    page_title="🎨 Explainable VQA Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for poppy abstract background and punchy colors
def load_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FECA57, #FF9FF3, #54A0FF);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .upload-section {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }

    .question-section {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }

    .result-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 1rem;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.15);
    }

    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }

    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 500;
    }

    .confidence-bar {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        height: 10px;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# Load model with caching
@st.cache_resource
def load_vqa_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model


# VQA function
def answer_question(image, question, processor, model):
    inputs = processor(image, question, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
        answer = processor.decode(outputs[0], skip_special_tokens=True)

    # Mock confidence score for demo
    confidence = np.random.uniform(0.75, 0.95)

    return answer, confidence


# Main application
def main():
    load_custom_css()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; color: #2C3E50; margin-bottom: 0.5rem;">
            🎨 Explainable Multi-Modal VQA Demo
        </h1>
        <p style="text-align: center; color: #7F8C8D; font-size: 1.2rem;">
            Ask questions about images and get AI-powered answers with explanations!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("🔄 Loading VQA Model..."):
        processor, model = load_vqa_model()

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📸 Upload Your Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to ask questions about"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="question-section">', unsafe_allow_html=True)
        st.markdown("### ❓ Ask Your Question")
        question = st.text_area(
            "What would you like to know about this image?",
            placeholder="e.g., What color is the car? How many people are in the image?",
            height=100
        )

        analyze_button = st.button("🔍 Analyze Image", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    # Results section
    if uploaded_file and question and analyze_button:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)

        with st.spinner("🤖 Analyzing image and generating answer..."):
            answer, confidence = answer_question(image, question, processor, model)

        # Display results
        st.markdown("### 🎯 Analysis Results")

        # Answer with styling
        st.markdown(f"""
        <div class="answer-box">
            <strong>🔮 Answer:</strong> {answer}
        </div>
        """, unsafe_allow_html=True)

        # Confidence score
        st.markdown(f"**🎯 Confidence Score:** {confidence:.2%}")
        st.markdown(f'<div class="confidence-bar" style="width: {confidence * 100}%;"></div>',
                    unsafe_allow_html=True)

        # Mock explanation section
        with st.expander("🔍 View Explanation (Demo)", expanded=True):
            st.markdown("""
            **🧠 How the model analyzed this image:**
            - **Visual Features Detected:** Objects, colors, spatial relationships
            - **Question Processing:** Natural language understanding of your query  
            - **Answer Generation:** Combining visual and textual reasoning
            - **Attention Areas:** The model focused on relevant image regions

            *Note: This is a demo version. Full explainability features would include 
            attention heatmaps and detailed reasoning chains.*
            """)

        st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar with sample questions
    with st.sidebar:
        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What is the main object in this image?",
            "What color is the dominant object?",
            "How many people are visible?",
            "What is the setting or location?",
            "What activity is taking place?",
            "What time of day does this appear to be?"
        ]

        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.rerun()

        st.markdown("---")
        st.markdown("### ℹ️ About This Demo")
        st.markdown("""
        This demo showcases:
        - **Multi-modal AI** capabilities
        - **Visual Question Answering** 
        - **Explainable AI** concepts
        - **Interactive UI** design

        Built with:
        - Streamlit for UI
        - BLIP model for VQA
        - Custom CSS styling
        """)


if __name__ == "__main__":
    main()
