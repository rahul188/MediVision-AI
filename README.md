# MedGemma 4B-IT Proof of Concept

A comprehensive Streamlit application demonstrating the capabilities of Google's MedGemma 4B-IT model for medical image analysis and clinical question answering.

## Features

🔍 **Medical Image Analysis**
- Chest X-ray interpretation
- Dermatology image assessment
- Histopathology analysis
- Ophthalmology image review

💬 **Medical Question Answering**
- Clinical knowledge queries
- Medical concept explanations
- Evidence-based responses

📝 **Medical Report Generation**
- Radiology reports
- Pathology reports
- Clinical summaries
- Discharge summaries

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Hugging Face account with MedGemma model access

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Hugging Face Authentication
Before using the model, you need to:
1. Create a Hugging Face account at https://huggingface.co
2. Visit https://huggingface.co/google/medgemma-4b-it
3. Accept the Health AI Developer Foundation's terms of use
4. Login to Hugging Face CLI:
```bash
huggingface-cli login
```

### 3. Run the Application
```bash
streamlit run app.py
```

## Model Information

**MedGemma 4B-IT** is a multimodal medical AI model developed by Google:
- **Size**: 4 billion parameters
- **Architecture**: Gemma 3 with SigLIP image encoder
- **Context Length**: 128K tokens
- **Training Data**: Medical images and text from various domains

### Performance Metrics
- MIMIC CXR: 88.9 F1 score
- MedQA: 87.7% accuracy
- SlakeVQA: 62.3 F1 score
- MedMCQA: 74.2% accuracy

## Usage Examples

### Image Analysis
Upload medical images or use sample images to get detailed medical analysis including:
- Anatomical descriptions
- Pathology identification
- Clinical impressions
- Diagnostic suggestions

### Question Answering
Ask medical questions and receive evidence-based answers covering:
- Disease information
- Treatment options
- Diagnostic procedures
- Medical concepts

### Report Generation
Generate professional medical reports based on clinical findings and patient information.

## Important Disclaimers

⚠️ **This is a research and educational tool only**
- Not intended for actual clinical diagnosis
- Always consult qualified healthcare professionals
- Outputs require clinical validation
- Not a substitute for medical expertise

## Technical Requirements

### Minimum System Requirements
- RAM: 8GB (16GB recommended)
- Storage: 10GB free space
- GPU: 4GB VRAM (for optimal performance)

### Recommended System Requirements
- RAM: 16GB or higher
- GPU: 8GB VRAM or higher
- Fast internet connection for model download

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure you have accepted the model terms on Hugging Face
   - Check your Hugging Face authentication
   - Verify sufficient disk space and memory

2. **CUDA Errors**
   - Install appropriate CUDA drivers
   - Check GPU memory availability
   - Fall back to CPU if necessary (slower)

3. **Memory Issues**
   - Close other applications
   - Use CPU mode for lower memory usage
   - Consider model quantization

## License

This project uses the MedGemma model which is governed by the Health AI Developer Foundations terms of use. Please review the terms at: https://developers.google.com/health-ai-developer-foundations/terms

## Support

For issues and questions:
- Check the troubleshooting section
- Review model documentation: https://developers.google.com/health-ai-developer-foundations/medgemma
- Visit the GitHub repository: https://github.com/google-health/medgemma

## Acknowledgments

- Google Health AI team for developing MedGemma
- Hugging Face for model hosting and transformers library
- Streamlit for the web application framework
