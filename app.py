import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
from PIL import Image
import requests
import io
import time
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Import logging utilities
from logger_utils import (
    PerformanceTimer, 
    medgemma_logger, 
    display_performance_metrics, 
    clear_performance_logs,
    time_operation
)

# Configure Streamlit page
st.set_page_config(
    page_title="MedGemma 4B-IT POC",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with light theme
st.markdown("""
<style>
    /* Force light theme styling */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    /* Main headers with better contrast */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    /* Info boxes with light theme */
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #000000;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #000000;
    }
    
    /* Ensure all text is dark on light background */
    .stMarkdown, .stText, p, div, span {
        color: #000000 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f5f5f5 !important;
    }
    
    /* Success/warning/error messages */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
    }
    
    .stWarning {
        background-color: #fff3cd !important;
        color: #856404 !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        color: #721c24 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Main header
st.markdown('<h1 class="main-header">🏥 MedGemma 4B-IT POC</h1>', unsafe_allow_html=True)

# GPU/CPU Status indicator at the top
device_available = "cuda" if torch.cuda.is_available() else "cpu"
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

col_status1, col_status2, col_status3 = st.columns([1, 2, 1])
with col_status2:
    if device_available == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown GPU"
        st.success(f"🚀 **GPU Acceleration Available**: {gpu_name}")
        st.info(f"📊 **PyTorch CUDA Version**: {torch.version.cuda} | **GPUs**: {gpu_count}")
    else:
        st.warning("💻 **Running on CPU** - GPU acceleration not available")
        st.info("💡 **Tip**: Install PyTorch with CUDA support for faster inference")

# Sidebar for model configuration
st.sidebar.header("🔧 Configuration")

# Device status in sidebar
st.sidebar.markdown("### 🖥️ System Status")
device_status = "cuda" if torch.cuda.is_available() else "cpu"
if device_status == "cuda":
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown GPU"
    st.sidebar.success(f"✅ GPU: {gpu_name}")
    st.sidebar.info(f"🔧 CUDA: {torch.version.cuda}")
    st.sidebar.info(f"📊 PyTorch: {torch.__version__}")
    
    # GPU memory info if available
    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.sidebar.info(f"💾 VRAM: {gpu_memory:.1f} GB")
    except:
        pass
else:
    st.sidebar.warning("⚠️ CPU Mode")
    st.sidebar.info(f"📊 PyTorch: {torch.__version__}")
    st.sidebar.info("💡 Install CUDA PyTorch for GPU acceleration")

# Model loading section
st.sidebar.markdown("### Model Status")
if st.session_state.model_loaded:
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.warning("⚠️ Model not loaded")

# Performance logging controls
st.sidebar.markdown("### 📊 Logging")
with st.sidebar.expander("Performance Settings"):
    st.write("**Current session stats:**")
    if 'performance_logs' in st.session_state and st.session_state.performance_logs:
        total_ops = len(st.session_state.performance_logs)
        total_time = sum(log['duration'] for log in st.session_state.performance_logs)
        st.write(f"• Operations: {total_ops}")
        st.write(f"• Total time: {total_time:.1f}s")
        st.write(f"• Avg time: {total_time/total_ops:.2f}s")
    else:
        st.write("• No operations yet")
    
    if st.button("📊 View Full Metrics", key="sidebar_metrics"):
        st.info("Check the Performance Monitor tab!")
    
    if st.button("🧹 Clear Logs", key="sidebar_clear"):
        clear_performance_logs()
        st.success("Logs cleared!")
        st.rerun()

# Load model button
if st.sidebar.button("🚀 Load MedGemma Model", type="primary"):
    with st.spinner("Loading MedGemma 4B-IT model... This may take a few minutes."):
        try:
            # Log system info on first load
            medgemma_logger.log_system_info()
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                st.sidebar.success(f"🚀 Loading on GPU: {gpu_name}")
            else:
                st.sidebar.info(f"💻 Loading on CPU: {device}")
            
            # Load the model and processor with timing
            model_id = "google/medgemma-4b-it"
            
            with PerformanceTimer(f"Model Loading: {model_id}"):
                medgemma_logger.log_model_loading(model_id, device)
                
                # Load with pipeline for easier use
                st.session_state.pipeline = pipeline(
                    "image-text-to-text",
                    model=model_id,
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    device=device if device == "cuda" else -1,
                )
                
                # Also load model and processor separately for more control
                st.session_state.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                )
                st.session_state.processor = AutoProcessor.from_pretrained(model_id)
            
            st.session_state.model_loaded = True
            st.sidebar.success("Model loaded successfully!")
            medgemma_logger.log_model_loaded(model_id, 0, None)  # Will be updated by timer
            st.rerun()
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            st.sidebar.error(error_msg)
            medgemma_logger.log_error("Model Loading", e)
            st.sidebar.info("Note: You may need to accept the model's terms of use on Hugging Face first.")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h2 class="section-header">📋 About MedGemma 4B-IT</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>MedGemma 4B-IT</strong> is a specialized medical AI model developed by Google that can:
    <ul>
        <li>🔍 Analyze medical images (X-rays, dermatology, pathology, ophthalmology)</li>
        <li>💬 Answer medical questions and provide clinical insights</li>
        <li>📊 Generate detailed medical reports</li>
        <li>🧠 Perform clinical reasoning tasks</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Model capabilities
    st.markdown("### 🎯 Key Capabilities")
    capabilities = [
        "Chest X-ray Analysis",
        "Dermatology Image Assessment",
        "Histopathology Analysis",
        "Ophthalmology Image Review",
        "Medical Question Answering",
        "Clinical Report Generation"
    ]
    
    for i, capability in enumerate(capabilities):
        st.write(f"{i+1}. {capability}")

with col2:
    st.markdown('<h2 class="section-header">📊 Model Performance</h2>', unsafe_allow_html=True)
    
    # Performance metrics visualization
    performance_data = {
        'Benchmark': ['MIMIC CXR F1', 'CheXpert F1', 'SlakeVQA F1', 'MedQA Accuracy', 'MedMCQA Accuracy'],
        'MedGemma 4B-IT': [88.9, 48.1, 62.3, 87.7, 74.2],
        'Base Gemma 3 4B': [81.1, 31.2, 38.6, 64.4, 55.7]
    }
    
    df = pd.DataFrame(performance_data)
    
    fig = px.bar(df, x='Benchmark', y=['MedGemma 4B-IT', 'Base Gemma 3 4B'],
                 title="Performance Comparison",
                 barmode='group',
                 color_discrete_map={'MedGemma 4B-IT': '#1f77b4', 'Base Gemma 3 4B': '#ff7f0e'})
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Interactive demo section
st.markdown('<h2 class="section-header">🧪 Interactive Demo</h2>', unsafe_allow_html=True)

if not st.session_state.model_loaded:
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Model Not Loaded</strong><br>
    Please load the MedGemma model using the sidebar to start the interactive demo.
    </div>
    """, unsafe_allow_html=True)
else:
    # Demo tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Analysis", "💬 Text Q&A", "📝 Report Generation", "📊 Performance Monitor"])
    
    with tab1:
        st.subheader("Medical Image Analysis")
        
        # Image upload options
        image_source = st.radio(
            "Choose image source:",
            ["Upload your own", "Use sample medical images"],
            horizontal=True
        )
        
        image = None
        
        if image_source == "Upload your own":
            uploaded_file = st.file_uploader(
                "Upload a medical image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload chest X-rays, dermatology images, or other medical scans"
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
        
        else:
            # Sample medical images
            sample_images = {
                "Chest X-ray": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
                "Sample Dermatology": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Melanoma.jpg/256px-Melanoma.jpg"
            }
            
            selected_sample = st.selectbox("Select a sample image:", list(sample_images.keys()))
            
            if st.button("Load Sample Image"):
                try:
                    response = requests.get(sample_images[selected_sample], 
                                          headers={"User-Agent": "MedGemma-POC"}, 
                                          stream=True)
                    image = Image.open(io.BytesIO(response.content))
                except Exception as e:
                    st.error(f"Error loading sample image: {str(e)}")
        
        if image:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Medical Image", use_container_width=True)
            
            with col2:
                # Analysis options
                analysis_type = st.selectbox(
                    "Analysis type:",
                    ["General Description", "Radiologist Review", "Detailed Findings", "Custom Prompt"]
                )
                
                prompts = {
                    "General Description": "Describe this medical image in detail.",
                    "Radiologist Review": "You are an expert radiologist. Analyze this image and provide your clinical assessment.",
                    "Detailed Findings": "Identify and describe any abnormalities, pathologies, or notable findings in this medical image."
                }
                
                if analysis_type == "Custom Prompt":
                    custom_prompt = st.text_area("Enter your custom prompt:", height=100)
                    prompt = custom_prompt if custom_prompt else "Describe this image."
                else:
                    prompt = prompts[analysis_type]
                
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Log analysis start
                            medgemma_logger.log_image_analysis_start(image.size, analysis_type)
                            
                            start_time = time.time()
                            
                            messages = [
                                {
                                    "role": "system",
                                    "content": [{"type": "text", "text": "You are an expert medical AI assistant."}]
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {"type": "image", "image": image}
                                    ]
                                }
                            ]
                            
                            with PerformanceTimer(f"Image Analysis: {analysis_type}"):
                                output = st.session_state.pipeline(text=messages, max_new_tokens=300)
                                result = output[0]["generated_text"][-1]["content"]
                            
                            duration = time.time() - start_time
                            token_count = len(result.split())  # Rough token estimate
                            
                            # Log completion
                            medgemma_logger.log_image_analysis_complete(analysis_type, duration, token_count)
                            
                            st.success("Analysis complete!")
                            st.markdown("### 📋 Analysis Result:")
                            st.markdown(result)
                            
                        except Exception as e:
                            error_msg = f"Error during analysis: {str(e)}"
                            st.error(error_msg)
                            medgemma_logger.log_error("Image Analysis", e)
    
    with tab2:
        st.subheader("Medical Question Answering")
        
        # Predefined medical questions
        sample_questions = [
            "What are the symptoms of pneumonia?",
            "Explain the difference between Type 1 and Type 2 diabetes.",
            "What are the risk factors for cardiovascular disease?",
            "Describe the stages of wound healing.",
            "What is the pathophysiology of asthma?"
        ]
        
        question_type = st.radio(
            "Question type:",
            ["Use sample questions", "Ask custom question"],
            horizontal=True
        )
        
        if question_type == "Use sample questions":
            selected_question = st.selectbox("Select a medical question:", sample_questions)
            question = selected_question
        else:
            question = st.text_area("Enter your medical question:", height=100)
        
        if question and st.button("💡 Get Answer", type="primary"):
            with st.spinner("Generating answer..."):
                try:
                    # Log Q&A start
                    medgemma_logger.log_text_qa_start(len(question))
                    
                    start_time = time.time()
                    
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are an expert medical AI assistant. Provide accurate, evidence-based medical information."}]
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": question}]
                        }
                    ]
                    
                    with PerformanceTimer(f"Text Q&A: {question[:50]}..."):
                        output = st.session_state.pipeline(text=messages, max_new_tokens=400)
                        result = output[0]["generated_text"][-1]["content"]
                    
                    duration = time.time() - start_time
                    token_count = len(result.split())  # Rough token estimate
                    
                    # Log completion
                    medgemma_logger.log_text_qa_complete(duration, len(result), token_count)
                    
                    st.success("Answer generated!")
                    st.markdown("### 💬 Answer:")
                    st.markdown(result)
                    
                except Exception as e:
                    error_msg = f"Error generating answer: {str(e)}"
                    st.error(error_msg)
                    medgemma_logger.log_error("Text Q&A", e)
    
    with tab3:
        st.subheader("Medical Report Generation")
        
        # Report generation options
        report_type = st.selectbox(
            "Report type:",
            ["Radiology Report", "Pathology Report", "Clinical Summary", "Discharge Summary"]
        )
        
        # Input fields for report generation
        patient_info = st.text_area(
            "Patient information / Clinical findings:",
            placeholder="Enter patient details, symptoms, examination findings, etc.",
            height=150
        )
        
        if patient_info and st.button("📝 Generate Report", type="primary"):
            with st.spinner("Generating medical report..."):
                try:
                    # Log report generation start
                    medgemma_logger.log_report_generation_start(report_type, len(patient_info))
                    
                    start_time = time.time()
                    
                    prompt = f"Generate a {report_type.lower()} based on the following information:\n\n{patient_info}"
                    
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": f"You are an expert medical professional. Generate a professional {report_type.lower()}."}]
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }
                    ]
                    
                    with PerformanceTimer(f"Report Generation: {report_type}"):
                        output = st.session_state.pipeline(text=messages, max_new_tokens=500)
                        result = output[0]["generated_text"][-1]["content"]
                    
                    duration = time.time() - start_time
                    token_count = len(result.split())  # Rough token estimate
                    
                    # Log completion
                    medgemma_logger.log_report_generation_complete(report_type, duration, len(result), token_count)
                    
                    st.success("Report generated!")
                    st.markdown(f"### 📄 {report_type}:")
                    st.markdown(result)
                    
                    # Option to download report
                    st.download_button(
                        label="📥 Download Report",
                        data=result,
                        file_name=f"{report_type.lower().replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    error_msg = f"Error generating report: {str(e)}"
                    st.error(error_msg)
                    medgemma_logger.log_error("Report Generation", e)
    
    with tab4:
        st.subheader("📊 Performance Monitoring")
        
        # Performance controls
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Refresh Metrics"):
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Logs"):
                clear_performance_logs()
                st.success("Performance logs cleared!")
                st.rerun()
        
        # Display performance metrics
        display_performance_metrics()
        
        # Real-time system monitoring
        st.subheader("🖥️ System Status")
        
        # Device information section
        st.markdown("#### 🚀 Compute Device Status")
        
        device_col1, device_col2 = st.columns(2)
        
        with device_col1:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                st.success(f"**GPU Available**: {gpu_name}")
                st.info(f"**CUDA Version**: {torch.version.cuda}")
                st.info(f"**PyTorch Version**: {torch.__version__}")
                
                # GPU temperature and utilization (if nvidia-ml-py is available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    st.metric("GPU Temperature", f"{temp}°C")
                    st.metric("GPU Utilization", f"{util.gpu}%")
                except:
                    st.info("💡 Install nvidia-ml-py for detailed GPU monitoring")
                    
            else:
                st.warning("**CPU Mode** - No GPU acceleration")
                st.info(f"**PyTorch Version**: {torch.__version__}")
                st.info("💡 Install CUDA-enabled PyTorch for GPU acceleration")
        
        with device_col2:
            # Current model device status
            if st.session_state.model_loaded and st.session_state.model is not None:
                try:
                    model_device = next(st.session_state.model.parameters()).device
                    st.success(f"**Model Device**: {model_device}")
                    
                    # Model memory usage
                    if torch.cuda.is_available() and 'cuda' in str(model_device):
                        allocated = torch.cuda.memory_allocated(0) / 1024**3
                        cached = torch.cuda.memory_reserved(0) / 1024**3
                        st.metric("Model VRAM", f"{allocated:.2f} GB allocated")
                        st.metric("Total VRAM", f"{cached:.2f} GB reserved")
                except:
                    st.info("Model device information unavailable")
            else:
                st.info("No model loaded")
        
        st.markdown("---")
        
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "CPU Usage", 
                    f"{cpu_percent}%",
                    delta=None
                )
            
            with col2:
                memory_used_gb = (memory.total - memory.available) / 1024**3
                memory_total_gb = memory.total / 1024**3
                st.metric(
                    "Memory Usage", 
                    f"{memory_used_gb:.1f}/{memory_total_gb:.1f} GB",
                    delta=f"{memory.percent}%"
                )
            
            with col3:
                # GPU status if available
                try:
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        st.metric(
                            "GPU Memory", 
                            f"{gpu_memory:.1f}/{gpu_total:.1f} GB",
                            delta=f"{(gpu_memory/gpu_total)*100:.1f}%"
                        )
                    else:
                        st.metric("GPU", "Not Available", delta=None)
                except:
                    st.metric("GPU", "Error", delta=None)
        
        except ImportError:
            st.warning("System monitoring requires psutil package")
        
        # Log file viewer
        st.subheader("📄 Recent Log Entries")
        
        try:
            import os
            log_dir = "logs"
            if os.path.exists(log_dir):
                log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
                if log_files:
                    latest_log = max(log_files)
                    log_path = os.path.join(log_dir, latest_log)
                    
                    with open(log_path, 'r', encoding='utf-8') as f:
                        log_lines = f.readlines()
                        recent_lines = log_lines[-20:]  # Last 20 lines
                        
                    st.text_area(
                        f"Latest log entries from {latest_log}:",
                        value=''.join(recent_lines),
                        height=300,
                        disabled=True
                    )
                else:
                    st.info("No log files found yet")
            else:
                st.info("Log directory not found")
        except Exception as e:
            st.warning(f"Could not read log files: {str(e)}")

# Footer with important disclaimers
st.markdown("---")
st.markdown("""
<div class="warning-box">
<strong>⚠️ Important Disclaimer</strong><br>
This is a proof of concept demonstration. The outputs are for educational and research purposes only. 
Always consult qualified healthcare professionals for medical advice and decisions. 
Do not use this tool for actual clinical diagnosis or treatment decisions.
</div>
""", unsafe_allow_html=True)

# Technical information
with st.expander("🔧 Technical Information"):
    st.markdown("""
    **Model Details:**
    - Model: google/medgemma-4b-it
    - Parameters: 4 billion
    - Architecture: Gemma 3 with SigLIP image encoder
    - Context Length: 128K tokens
    - Supported Image Types: Medical images (X-rays, dermatology, pathology, ophthalmology)
    
    **Performance:**
    - MIMIC CXR: 88.9 F1 score
    - MedQA: 87.7% accuracy
    - SlakeVQA: 62.3 F1 score
    
    **Requirements:**
    - GPU recommended for optimal performance
    - Hugging Face account with model access approval
    """)
