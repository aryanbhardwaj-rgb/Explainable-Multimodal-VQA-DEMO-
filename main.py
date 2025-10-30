import streamlit as st
import time
import random
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="🎨 Explainable VQA Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_animated_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FECA57, #FF9FF3, #54A0FF);
        background-size: 400% 400%;
        animation: gradientShift 10s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        25% { background-position: 100% 50%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }

    .main-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeInDown 1s ease-out;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .upload-section {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInLeft 0.8s ease-out;
        border: 2px dashed transparent;
    }

    .upload-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        border-color: #FF6B6B;
        background: rgba(255, 255, 255, 0.95);
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .question-section {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInRight 0.8s ease-out;
    }

    .question-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 0.95);
    }

    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .stTextArea textarea {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.8);
    }

    .stTextArea textarea:focus {
        border-color: #FF6B6B;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
        transform: scale(1.02);
        background: rgba(255, 255, 255, 1);
    }

    .result-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 1rem;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.15);
        animation: fadeInUp 0.6s ease-out;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2.5rem;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        background: linear-gradient(45deg, #4ECDC4, #FF6B6B);
    }

    .stButton > button:active {
        transform: translateY(-1px) scale(1.02);
    }

    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 500;
        animation: pulse 2s infinite;
        position: relative;
        overflow: hidden;
    }

    .answer-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    .confidence-bar {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        height: 12px;
        border-radius: 6px;
        margin: 0.5rem 0;
        animation: fillBar 2s ease-out;
        position: relative;
        overflow: hidden;
    }

    @keyframes fillBar {
        from { width: 0%; }
        to { width: var(--confidence-width); }
    }

    .confidence-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 30%;
        background: rgba(255, 255, 255, 0.3);
        animation: slide 2s infinite;
        border-radius: 6px;
    }

    @keyframes slide {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(400%); }
    }

    .loading-dots {
        display: inline-block;
    }

    .loading-dots span {
        animation: loadingDot 1.4s infinite both;
    }

    .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
    .loading-dots span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes loadingDot {
        0%, 80%, 100% { opacity: 0; }
        40% { opacity: 1; }
    }

    .image-upload-area {
        border: 3px dashed #ccc;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        background: linear-gradient(45deg, rgba(255, 107, 107, 0.1), rgba(78, 205, 196, 0.1));
    }

    .image-upload-area:hover {
        border-color: #FF6B6B;
        background: linear-gradient(45deg, rgba(255, 107, 107, 0.2), rgba(78, 205, 196, 0.2));
        transform: scale(1.02);
    }

    .typewriter {
        overflow: hidden;
        border-right: 3px solid #FF6B6B;
        white-space: nowrap;
        margin: 0 auto;
        animation: typing 2s steps(40, end), blink-caret 0.75s step-end infinite;
    }

    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent; }
        50% { border-color: #FF6B6B; }
    }
    </style>
    """, unsafe_allow_html=True)


def get_mock_vqa_response(question, has_image=False):
    responses = {
        "color": [
            "The dominant color in this image is blue.",
            "I can see red and white colors prominently.",
            "The main colors are green and brown.",
            "There are multiple colors including yellow, orange, and purple."
        ],
        "people": [
            "I can see 2 people in this image.",
            "There are 3 people visible in the scene.",
            "I detect 1 person in the image.",
            "There appear to be 4 people in this photo."
        ],
        "object": [
            "The main object in this image is a car.",
            "I can see a large building as the primary subject.",
            "The central object appears to be a bicycle.",
            "The main focus is on a beautiful tree."
        ],
        "location": [
            "This appears to be taken in a park or outdoor setting.",
            "The location looks like an indoor office or workspace.",
            "This seems to be a residential area with houses.",
            "The setting appears to be a busy city street."
        ],
        "activity": [
            "The people in the image appear to be walking.",
            "I can see someone riding a bicycle.",
            "The activity shown is people having a conversation.",
            "The scene shows people working at computers."
        ]
    }

    question_lower = question.lower()

    if any(word in question_lower for word in ["color", "colour"]):
        category = "color"
    elif any(word in question_lower for word in ["people", "person", "many", "how many"]):
        category = "people"
    elif any(word in question_lower for word in ["object", "thing", "main", "what is"]):
        category = "object"
    elif any(word in question_lower for word in ["where", "location", "place", "setting"]):
        category = "location"
    elif any(word in question_lower for word in ["doing", "activity", "action", "happening"]):
        category = "activity"
    else:
        category = random.choice(list(responses.keys()))

    response = random.choice(responses[category])
    confidence = random.uniform(0.78, 0.95)

    return response, confidence


def typewriter_effect(text, container):
    displayed_text = ""
    for char in text:
        displayed_text += char
        container.markdown(f'<div class="typewriter">{displayed_text}</div>', unsafe_allow_html=True)
        time.sleep(0.05)


def main():
    load_animated_css()

    st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; color: #2C3E50; margin-bottom: 0.5rem;">
            🎨 Explainable Multi-Modal VQA Demo
        </h1>
        <p style="text-align: center; color: #7F8C8D; font-size: 1.2rem;">
            Ask questions about images and get AI-powered answers with explanations!
        </p>
        <p style="text-align: center; color: #e74c3c; font-size: 1rem; font-weight: bold;">
            ✨ Demo Mode - Mock Responses Enabled ✨
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📸 Upload Your Image")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to ask questions about",
            key="image_uploader"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="✅ Uploaded Successfully!", use_column_width=True)

            st.markdown("""
            <div style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4); 
                        color: white; padding: 10px; border-radius: 10px; 
                        text-align: center; margin-top: 10px;
                        animation: fadeIn 1s ease-out;">
                📊 Image loaded and ready for analysis!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="image-upload-area">
                <h3>🖼️ Drag & Drop Your Image Here</h3>
                <p>Supported formats: JPG, JPEG, PNG, BMP</p>
                <div style="font-size: 48px; margin: 20px;">📁</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="question-section">', unsafe_allow_html=True)
        st.markdown("### ❓ Ask Your Question")

        question = st.text_area(
            "What would you like to know about this image?",
            placeholder="e.g., What color is the car? How many people are in the image?",
            height=120,
            key="question_input"
        )

        analyze_button = st.button(
            "🔍 Analyze Image",
            type="primary",
            key="analyze_btn"
        )

        st.markdown("#### 💡 Quick Questions:")
        quick_questions = [
            "What is the main object?",
            "How many people are there?",
            "What colors do you see?",
            "What is the setting?"
        ]

        cols = st.columns(2)
        for i, q in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(q, key=f"quick_{i}"):
                    st.session_state.question_input = q
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    if question and analyze_button:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)

        with st.spinner(""):
            st.markdown("""
            <div style="text-align: center; margin: 20px;">
                <h3>🤖 AI is analyzing your image
                <span class="loading-dots">
                    <span>.</span><span>.</span><span>.</span>
                </span>
                </h3>
            </div>
            """, unsafe_allow_html=True)

            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)

            answer, confidence = get_mock_vqa_response(question, uploaded_file is not None)

        st.markdown("### 🎯 Analysis Results")

        st.markdown(f"""
        <div class="answer-box">
            <strong>🔮 Answer:</strong> {answer}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**🎯 Confidence Score:** {confidence:.2%}")
        confidence_width = confidence * 100
        st.markdown(f'''
        <div class="confidence-bar" style="--confidence-width: {confidence_width}%;"></div>
        ''', unsafe_allow_html=True)

        with st.expander("🔍 View Detailed Explanation", expanded=True):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        color: white; padding: 1.5rem; border-radius: 10px; margin: 10px 0;">
                <h4>🧠 AI Analysis Process:</h4>
            </div>
            """, unsafe_allow_html=True)

            explanation_steps = [
                "🔍 **Image Processing:** Analyzing visual features and objects",
                "📝 **Question Understanding:** Processing natural language query",
                "🤝 **Multi-modal Fusion:** Combining visual and textual information",
                "💡 **Answer Generation:** Producing contextual response",
                "📊 **Confidence Calculation:** Estimating prediction reliability"
            ]

            for i, step in enumerate(explanation_steps):
                time.sleep(0.3)
                st.markdown(step)
                if i < len(explanation_steps) - 1:
                    st.markdown("↓")

            st.info(
                "💼 **Demo Note:** This is a demonstration version with mock responses. In the full version, this would include attention heatmaps and detailed reasoning chains.")

        st.markdown('</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🚀 Demo Features")

        features = [
            "✨ Animated UI Elements",
            "🎨 Dynamic Gradient Backgrounds",
            "📱 Responsive Design",
            "🤖 Mock VQA Responses",
            "📊 Confidence Visualization",
            "💡 Quick Question Suggestions"
        ]

        for feature in features:
            st.markdown(f"- {feature}")

        st.markdown("---")

        st.markdown("### 🎯 Sample Questions")
        sample_questions = [
            "What is the main object in this image?",
            "What color is the dominant object?",
            "How many people are visible?",
            "What is the setting or location?",
            "What activity is taking place?",
            "What time of day does this appear to be?"
        ]

        for q in sample_questions:
            if st.button(q, key=f"sidebar_{q}"):
                st.session_state.question_input = q
                st.rerun()

        st.markdown("---")
        st.markdown("### ℹ️ About This Demo")
        st.markdown("""
        **Current Features:**
        - ✅ Animated UI elements
        - ✅ Mock VQA responses  
        - ✅ Interactive components
        - ✅ Confidence scoring

        **Coming Soon:**
        - 🔄 Real VQA model integration
        - 🎯 Attention heatmaps
        - 📈 Detailed explanations
        - 🔍 Advanced analytics
        """)


if __name__ == "__main__":
    main()
