import streamlit as st
from PIL import Image
from Utils.classification import load_classification_model, classify_image
from Utils.segment import load_segmentation_model, segment_image
from Utils.report import generate_pdf_report
import base64

# -----------------------------
# Page Config with Logo/Favicon
# -----------------------------

def get_base64_encoded_image(image_path):
    """Convert image to base64 for favicon"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Load favicon if available
favicon_b64 = get_base64_encoded_image("favicon.ico")

st.set_page_config(
    page_title="TumorX - Brain Tumor AI Analysis",
    page_icon=f"data:image/png;base64,{favicon_b64}" if favicon_b64 else "🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Load Models Once
# -----------------------------
@st.cache_resource
def load_models():
    cls_model = load_classification_model("models/brain_tumor_model.keras")
    seg_model = load_segmentation_model("models/final_model.keras")
    return cls_model, seg_model

cls_model, seg_model = load_models()

# -----------------------------
# Enhanced Custom CSS with Dark Theme
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Orbitron:wght@400;500;600;700;800;900&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* Remove top spacing */
    .stApp > div:first-child,
    .main .block-container,
    section.main > div,
    .element-container:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #000000 100%);
        background-attachment: fixed;
        min-height: 100vh;
        color: #ffffff;
        padding-top: 0 !important;
        margin-top: 0 !important;
        position: relative;
        overflow-x: hidden;
    }

    /* Animated background particles */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, rgba(59, 130, 246, 0.08) 0%, transparent 50%);
        animation: floatingParticles 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }

    @keyframes floatingParticles {
        0%, 100% { transform: translateX(0) translateY(0); }
        25% { transform: translateX(-20px) translateY(-10px); }
        50% { transform: translateX(20px) translateY(-20px); }
        75% { transform: translateX(-10px) translateY(10px); }
    }

    /* Interactive Logo Header */
    .logo-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        animation: slideDown 1.5s ease-out;
    }

    .logo-header::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translateX(-50%) translateY(-50%);
        width: 200px;
        height: 200px;
        background: conic-gradient(
            from 0deg,
            rgba(99, 102, 241, 0.1) 0deg,
            rgba(139, 92, 246, 0.15) 60deg,
            rgba(59, 130, 246, 0.1) 120deg,
            rgba(99, 102, 241, 0.05) 180deg,
            rgba(139, 92, 246, 0.1) 240deg,
            rgba(99, 102, 241, 0.15) 300deg,
            rgba(99, 102, 241, 0.1) 360deg
        );
        border-radius: 50%;
        z-index: 0;
        animation: rotate 8s linear infinite, breathe 4s ease-in-out infinite alternate;
    }

    @keyframes rotate {
        from { transform: translateX(-50%) translateY(-50%) rotate(0deg); }
        to { transform: translateX(-50%) translateY(-50%) rotate(360deg); }
    }

    @keyframes breathe {
        from { scale: 1; opacity: 0.4; }
        to { scale: 1.2; opacity: 0.7; }
    }

    @keyframes slideDown {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    .logo-container {
        position: relative;
        z-index: 1;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-block;
    }

    .logo-container:hover {
        transform: translateY(-10px) scale(1.05);
        filter: drop-shadow(0 15px 30px rgba(99,102,241,0.5));
    }

    .logo-title {
        font-family: 'Orbitron', monospace;
        font-size: clamp(3rem, 8vw, 5rem);
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: float 6s ease-in-out infinite;
        margin: 0;
        text-shadow: 0 0 30px rgba(99,102,241,0.3);
    }

    .logo-subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 300;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        25% { transform: translateY(-8px); }
        50% { transform: translateY(0); }
        75% { transform: translateY(-4px); }
    }

    /* File Upload Enhancement */
    .upload-container {
        max-width: 700px;
        margin: 3rem auto;
        animation: fadeInUp 1s ease-out 0.5s both;
    }

    .stFileUploader {
        border: none !important;
        background: none !important;
    }

    .stFileUploader > div {
        border: 2px dashed rgba(99, 102, 241, 0.4) !important;
        border-radius: 25px !important;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%) !important;
        backdrop-filter: blur(15px) !important;
        padding: 4rem 2rem !important;
        transition: all 0.4s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stFileUploader > div:hover {
        border-color: rgba(99, 102, 241, 0.7) !important;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.25) !important;
    }

    .stFileUploader label {
        color: #e5e7eb !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 1rem !important;
        cursor: pointer !important;
    }

    .stFileUploader label::before {
        content: "🧠";
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
    }

    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-15px); }
        60% { transform: translateY(-8px); }
    }

    /* Results Container */
    .results-container {
        animation: fadeInUp 1s ease-out 0.8s both;
        margin-top: 3rem;
    }

    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0 1.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Image Display Cards */
    .image-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.04) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }

    .image-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99,102,241,0.5);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3), 0 0 20px rgba(99,102,241,0.2);
    }

    .image-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #d946ef);
        background-size: 200% 100%;
        animation: gradientMove 3s linear infinite;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    .image-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f8fafc;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Prediction Results Card */
    .prediction-container {
        max-width: 600px;
        margin: 3rem auto;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(99,102,241,0.3);
        border-radius: 25px;
        padding: 2.5rem;
        text-align: center;
        animation: scaleIn 0.8s ease-out, borderGlow 4s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }

    @keyframes scaleIn {
        from { transform: scale(0.9); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }

    @keyframes borderGlow {
        0%, 100% {
            border-color: rgba(99,102,241,0.3);
            box-shadow: 0 0 20px rgba(99,102,241,0.2);
        }
        50% {
            border-color: rgba(99,102,241,0.6);
            box-shadow: 0 0 30px rgba(99,102,241,0.4);
        }
    }

    .prediction-title {
        font-size: 1.4rem;
        color: #d1d5db;
        margin-bottom: 2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .prediction-result {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 1.5rem 0;
        text-shadow: 0 0 20px currentColor;
        animation: textPulse 2s ease-in-out infinite;
    }

    @keyframes textPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    .confidence-score {
        font-size: 1.5rem;
        font-weight: 600;
        color: #60a5fa;
        margin: 1rem 0;
        animation: countUp 2s ease-out;
    }

    @keyframes countUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Generate Report Button Enhancement */
    .report-container {
        text-align: center;
        margin: 3rem 0 2rem;
        animation: fadeInUp 1s ease-out 1.5s both;
    }

    .stButton > button {
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 1rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 10px 25px rgba(16, 185, 129, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        cursor: pointer !important;
        min-width: 280px !important;
        height: 60px !important;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #047857 0%, #059669 50%, #10b981 100%) !important;
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 
            0 15px 35px rgba(16, 185, 129, 0.4),
            0 5px 15px rgba(16, 185, 129, 0.3) !important;
    }

    /* Download Button Enhancement */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 50%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 0.8rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.3) !important;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 25px rgba(124, 58, 237, 0.4) !important;
        background: linear-gradient(135deg, #6d28d9 0%, #7c3aed 50%, #8b5cf6 100%) !important;
    }

    /* Section spacing and animations */
    @keyframes fadeInUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* Image enhancements */
    .stImage img {
        border-radius: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }

    .stImage:hover img {
        transform: scale(1.02);
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    }

    /* Subheader styling */
    h3 {
        color: #a5b4fc !important;
        font-weight: 700 !important;
        text-align: center !important;
        margin: 2rem 0 1rem !important;
        font-size: 1.6rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }

    /* Text color improvements */
    .stMarkdown p {
        color: #e5e7eb !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }

    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 15px !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        animation: slideInLeft 0.5s ease-out !important;
    }

    @keyframes slideInLeft {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    /* Spinner enhancement */
    .stSpinner {
        background: rgba(0,0,0,0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
    }

    .stSpinner > div {
        border-color: #6366f1 transparent #8b5cf6 transparent !important;
        animation: spin 1s linear infinite, colorShift 2s ease-in-out infinite !important;
    }

    @keyframes colorShift {
        0%, 100% { border-top-color: #6366f1; border-bottom-color: #8b5cf6; }
        50% { border-top-color: #8b5cf6; border-bottom-color: #d946ef; }
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .logo-title {
            font-size: 2.5rem;
        }
        
        .prediction-container {
            margin: 2rem 1rem;
            padding: 2rem;
        }
        
        .prediction-result {
            font-size: 2rem;
        }

        .stButton > button {
            min-width: 240px !important;
            font-size: 1.1rem !important;
            padding: 0.9rem 2.5rem !important;
        }
    }

    /* Column spacing */
    .stColumn {
        padding: 0 0.75rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Interactive Logo Header
# -----------------------------
st.markdown(
    """
    <div class="logo-header">
        <div class="logo-container" onclick="
            this.style.animation='none';
            this.offsetHeight;
            this.style.animation='float 6s ease-in-out infinite, scaleIn 0.8s ease-out';
        ">
            <div class="logo-title">TumorX</div>
            <div class="logo-subtitle">AI Brain Tumor Analysis</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# File Upload Section
# -----------------------------
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "🧠 Upload MRI Scan for AI Analysis", 
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Main Analysis Section
# -----------------------------
if uploaded_file is not None:
    with st.spinner('🔄 Analyzing MRI scan with advanced AI models...'):
        # Load and process image
        image = Image.open(uploaded_file)
        
        # Classification
        class_label, confidence = classify_image(cls_model, image)
        
        # Segmentation
        try:
            segmented_img = segment_image(seg_model, image)
        except Exception as e:
            st.warning(f"⚠️ Segmentation analysis unavailable: {str(e)}")
            segmented_img = None

    # Results Section
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    # Display Images Side by Side
    st.markdown('<h3 class="section-header">📊 Image Analysis Results</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown('<div class="image-title">🔬 Original MRI Scan</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        if segmented_img is not None:
            st.markdown('<div class="image-title">🎯 AI Segmentation Analysis</div>', unsafe_allow_html=True)
            st.image(segmented_img, use_container_width=True)
        else:
            st.markdown('<div class="image-title">⚠️ Segmentation Unavailable</div>', unsafe_allow_html=True)
            st.info("Segmentation analysis could not be performed on this image.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Classification Results
    result_color = "#ef4444" if class_label != "notumor" else "#10b981"
    result_emoji = "⚠️" if class_label != "notumor" else "✅"
    
    st.markdown(
        f"""
        <div class="prediction-container">
            <div class="prediction-title">🩺 AI DIAGNOSTIC ANALYSIS</div>
            <div class="prediction-result" style="color: {result_color};">
                {result_emoji} {class_label.upper()}
            </div>
            <div class="confidence-score">
                Model Confidence: {confidence*100:.1f}%
            </div>
            <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(0,0,0,0.3); border-radius: 15px; font-size: 1.1rem; color: #e5e7eb;">
                <strong>{'⚠️ Consult medical professional for further evaluation' if class_label != 'notumor' else '✅ Scan appears normal - No tumor detected'}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Generate Report Section
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📄 Generate Patient Report</h3>', unsafe_allow_html=True)
    
    if st.button("📑 Generate PDF Report"):
        try:
            pdf_path = generate_pdf_report(class_label, confidence, image, segmented_img)
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Medical Report",
                    data=f,
                    file_name="TumorX_Medical_Report.pdf",
                    mime="application/pdf"
                )
            st.success("✅ Report generated successfully!")
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer Information
# -----------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(30, 41, 59, 0.6) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 4rem 1rem 2rem;
        animation: fadeInUp 1s ease-out 1.8s both;
        text-align: center;
    ">
        <div style="font-size: 1.3rem; font-weight: 700; color: #a5b4fc; margin-bottom: 1rem;">
            🧬 About TumorX AI Platform
        </div>
        <div style="font-size: 1rem; line-height: 1.7; color: #d1d5db; max-width: 800px; margin: 0 auto;">
            TumorX leverages state-of-the-art deep learning models for accurate brain tumor detection and segmentation.
            Our AI system provides rapid, reliable analysis to support medical professionals in diagnostic workflows.
            <br><br>
            <strong style="color: #fbbf24;">⚠️ MEDICAL DISCLAIMER:</strong> This AI system is for research and educational purposes only. 
            Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.
            <br><br>
            <em style="color: #94a3b8;">⚡ Powered by TensorFlow • Deep Learning Excellence • Healthcare Innovation</em>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)