"""
Utility functions for MedGemma POC
"""

import torch
import streamlit as st
from PIL import Image
import requests
import io
from typing import Optional, Tuple, Dict, Any

def check_gpu_availability() -> Tuple[bool, str]:
    """
    Check if GPU is available and return status
    
    Returns:
        Tuple of (gpu_available, device_info)
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        return True, f"{gpu_name} ({gpu_count} device(s))"
    else:
        return False, "CPU only"

def load_sample_image(url: str) -> Optional[Image.Image]:
    """
    Load a sample image from URL
    
    Args:
        url: Image URL
        
    Returns:
        PIL Image or None if failed
    """
    try:
        headers = {"User-Agent": "MedGemma-POC/1.0"}
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
        return None

def format_medical_response(response: str) -> str:
    """
    Format medical AI response for better readability
    
    Args:
        response: Raw model response
        
    Returns:
        Formatted response
    """
    # Add some basic formatting
    lines = response.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            # Add bullet points for lists
            if line.lower().startswith(('- ', '• ', '* ')):
                formatted_lines.append(line)
            elif any(line.lower().startswith(prefix) for prefix in ['findings:', 'impression:', 'recommendation:', 'diagnosis:']):
                formatted_lines.append(f"**{line}**")
            else:
                formatted_lines.append(line)
    
    return '\n\n'.join(formatted_lines)

def get_prompt_templates() -> Dict[str, str]:
    """
    Get predefined prompt templates for different medical tasks
    
    Returns:
        Dictionary of prompt templates
    """
    return {
        "chest_xray": "You are an expert radiologist. Analyze this chest X-ray and provide a detailed assessment including: 1) Technical quality, 2) Anatomical structures visible, 3) Any abnormalities or pathologies, 4) Clinical impression.",
        
        "dermatology": "You are an expert dermatologist. Examine this skin lesion/condition and provide: 1) Description of the lesion, 2) Differential diagnosis, 3) Recommended next steps or referrals.",
        
        "pathology": "You are an expert pathologist. Analyze this histopathological image and provide: 1) Tissue type and staining, 2) Cellular morphology, 3) Any pathological changes, 4) Diagnostic impression.",
        
        "ophthalmology": "You are an expert ophthalmologist. Examine this fundus/retinal image and assess: 1) Overall retinal appearance, 2) Optic disc evaluation, 3) Vascular changes, 4) Any pathological findings.",
        
        "general_medical": "You are an expert medical AI assistant. Provide accurate, evidence-based information while being clear that this is for educational purposes only and not a substitute for professional medical advice."
    }

def validate_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Validate uploaded image for medical analysis
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (is_valid, message)
    """
    if image is None:
        return False, "No image provided"
    
    # Check image size
    width, height = image.size
    if width < 100 or height < 100:
        return False, "Image too small (minimum 100x100 pixels)"
    
    if width > 4000 or height > 4000:
        return False, "Image too large (maximum 4000x4000 pixels)"
    
    # Check file size (approximate)
    if hasattr(image, 'size'):
        # Estimate file size
        estimated_size = width * height * 3  # RGB
        if estimated_size > 50 * 1024 * 1024:  # 50MB
            return False, "Image file too large"
    
    return True, "Image valid"

def get_medical_disclaimers() -> Dict[str, str]:
    """
    Get medical disclaimers for different contexts
    
    Returns:
        Dictionary of disclaimer texts
    """
    return {
        "general": "⚠️ This AI tool is for educational and research purposes only. Always consult qualified healthcare professionals for medical advice.",
        
        "diagnosis": "⚠️ This analysis is not a medical diagnosis. Clinical correlation and professional medical evaluation are required.",
        
        "research": "⚠️ This is a research prototype. Results should not be used for clinical decision-making without proper validation.",
        
        "emergency": "🚨 For medical emergencies, contact emergency services immediately. Do not rely on AI tools for urgent medical situations."
    }

def create_medical_report_template(report_type: str) -> str:
    """
    Create template for medical reports
    
    Args:
        report_type: Type of medical report
        
    Returns:
        Report template
    """
    templates = {
        "radiology": """
RADIOLOGY REPORT

Patient ID: [To be filled]
Study Date: [To be filled]
Study Type: [To be filled]

CLINICAL INDICATION:
[Clinical indication]

TECHNIQUE:
[Imaging technique and parameters]

FINDINGS:
[Detailed findings]

IMPRESSION:
[Radiological impression]

Reporting Radiologist: AI Assistant (Educational Use Only)
        """,
        
        "pathology": """
PATHOLOGY REPORT

Patient ID: [To be filled]
Specimen: [Specimen type]
Date: [Date]

CLINICAL HISTORY:
[Clinical history]

GROSS DESCRIPTION:
[Gross findings]

MICROSCOPIC EXAMINATION:
[Microscopic findings]

DIAGNOSIS:
[Pathological diagnosis]

Pathologist: AI Assistant (Educational Use Only)
        """,
        
        "clinical": """
CLINICAL SUMMARY

Patient: [Patient information]
Date: [Date]

CHIEF COMPLAINT:
[Primary concern]

HISTORY OF PRESENT ILLNESS:
[Current illness history]

PHYSICAL EXAMINATION:
[Examination findings]

ASSESSMENT AND PLAN:
[Clinical assessment and treatment plan]

Provider: AI Assistant (Educational Use Only)
        """
    }
    
    return templates.get(report_type, templates["clinical"])

# Cache commonly used data
@st.cache_data
def get_sample_medical_images() -> Dict[str, str]:
    """
    Get URLs for sample medical images
    
    Returns:
        Dictionary of sample image URLs
    """
    return {
        "Chest X-ray (Normal)": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
        "Chest X-ray (Pneumonia)": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Pneumonia_x-ray.jpg/256px-Pneumonia_x-ray.jpg",
        "Skin Lesion": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Melanoma.jpg/256px-Melanoma.jpg"
    }

@st.cache_data
def get_sample_medical_questions() -> list:
    """
    Get sample medical questions for testing
    
    Returns:
        List of sample questions
    """
    return [
        "What are the symptoms and treatment options for pneumonia?",
        "Explain the difference between Type 1 and Type 2 diabetes.",
        "What are the risk factors for cardiovascular disease?",
        "Describe the pathophysiology of asthma.",
        "What are the stages of wound healing?",
        "Explain the mechanism of action of beta-blockers.",
        "What are the signs and symptoms of a myocardial infarction?",
        "Describe the different types of fractures and their treatment.",
        "What is the role of inflammation in autoimmune diseases?",
        "Explain the process of blood coagulation."
    ]
