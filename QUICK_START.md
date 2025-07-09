# MedGemma 4B-IT POC - Quick Start Guide

## 🚀 Quick Setup

### Step 1: Open Terminal in VS Code
Press `Ctrl+Shift+`` or go to Terminal → New Terminal

### Step 2: Navigate to Project Directory
```powershell
cd "d:\llm-lab\MED_GEMMA"
```

### Step 3: Run Setup (Choose One Method)

#### Method A: Automated Setup (Recommended)
```powershell
.\setup.bat
```

#### Method B: Manual Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Step 4: Setup Hugging Face Authentication
1. Create account at https://huggingface.co
2. Visit https://huggingface.co/google/medgemma-4b-it
3. Accept the Health AI Developer Foundation's terms
4. Get your token from https://huggingface.co/settings/tokens
5. Login via CLI:
```powershell
huggingface-cli login
```

### Step 5: Run the Application
```powershell
# If using automated setup
.\run.bat

# If using manual setup
streamlit run app.py
```

## 🔧 System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 10GB free disk space
- Internet connection

### Recommended Requirements
- Python 3.9+
- 16GB+ RAM
- CUDA-compatible GPU (8GB+ VRAM)
- Fast internet for model download

## 📋 Features Overview

### 🖼️ Medical Image Analysis
- **Chest X-rays**: Automated interpretation and findings
- **Dermatology**: Skin lesion analysis and assessment
- **Pathology**: Histopathological image review
- **Ophthalmology**: Fundus and retinal image analysis

### 💬 Medical Q&A
- Clinical knowledge queries
- Disease information
- Treatment explanations
- Medical concept clarification

### 📝 Report Generation
- Radiology reports
- Pathology reports
- Clinical summaries
- Discharge summaries

## 🎯 Usage Examples

### Example 1: Chest X-ray Analysis
1. Load the MedGemma model (sidebar)
2. Go to "Image Analysis" tab
3. Upload chest X-ray or use sample
4. Select "Radiologist Review"
5. Click "Analyze Image"

### Example 2: Medical Question
1. Go to "Text Q&A" tab
2. Select sample question or enter custom
3. Click "Get Answer"

### Example 3: Generate Report
1. Go to "Report Generation" tab
2. Select report type
3. Enter patient information/findings
4. Click "Generate Report"

## ⚠️ Important Notes

### Medical Disclaimer
- **Educational/Research Use Only**
- Not for actual clinical diagnosis
- Always consult healthcare professionals
- Validate all outputs clinically

### First-Time Usage
- Model download: ~9GB (one-time)
- First load: 2-5 minutes
- Subsequent loads: 30-60 seconds

### Performance Tips
- **GPU Recommended**: 4x faster than CPU
- **Memory**: Close other applications
- **Network**: Stable connection for model download

## 🛠️ Troubleshooting

### Common Issues

#### "Model Loading Failed"
- Check Hugging Face authentication
- Verify terms acceptance
- Ensure sufficient disk space
- Try restarting application

#### "CUDA Out of Memory"
- Close other GPU applications
- Use CPU mode (slower)
- Restart the application

#### "Import Errors"
- Activate virtual environment
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version compatibility

#### "Authentication Failed"
- Re-run: `huggingface-cli login`
- Check token validity
- Verify model access approval

### Getting Help
1. Check this guide first
2. Review error messages carefully
3. Ensure all prerequisites are met
4. Check system resources (RAM, disk, GPU)

## 📚 Additional Resources

- **Model Documentation**: https://developers.google.com/health-ai-developer-foundations/medgemma
- **Hugging Face Model Page**: https://huggingface.co/google/medgemma-4b-it
- **GitHub Repository**: https://github.com/google-health/medgemma
- **Quick Start Notebook**: [Colab Link](https://colab.research.google.com/github/google-health/medgemma/blob/main/notebooks/quick_start_with_hugging_face.ipynb)

## 🔄 Next Steps

After successful setup:
1. Explore all three main features
2. Try different types of medical images
3. Experiment with various question types
4. Generate different report formats
5. Consider fine-tuning for specific use cases

## 📞 Support

For technical issues:
- Review troubleshooting section
- Check system requirements
- Verify all setup steps completed
- Consult official documentation

---

**Happy exploring with MedGemma! 🏥🤖**
