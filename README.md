# 🏥 MediVision-AI

**Advanced Medical AI Assistant powered by Google's MedGemma 4B-IT**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Overview

MediVision-AI is a cutting-edge medical AI assistant that leverages Google's MedGemma 4B-IT model to provide:

- 🔍 **Medical Image Analysis** - Analyze X-rays, dermatology images, pathology slides, and ophthalmology scans
- 💬 **Medical Q&A** - Get evidence-based answers to medical questions
- 📝 **Report Generation** - Generate professional medical reports (radiology, pathology, clinical summaries)
- 📊 **Performance Monitoring** - Real-time system and GPU monitoring

## ✨ Features

### �️ Image Analysis
- **Chest X-ray Analysis** with detailed findings
- **Dermatology Assessment** for skin conditions
- **Histopathology Analysis** for tissue examination
- **Ophthalmology Review** for eye-related conditions

### 🧠 AI Capabilities
- **Clinical Reasoning** based on medical evidence
- **Multi-modal Processing** (text + images)
- **Context-aware Responses** with medical knowledge
- **Professional Report Generation**

### ⚡ Performance
- **GPU Acceleration** with CUDA support
- **Real-time Monitoring** of system resources
- **Performance Metrics** tracking
- **Optimized Inference** with PyTorch

## 🏆 Model Performance

| Benchmark | MediVision-AI (MedGemma 4B-IT) | Base Gemma 3 4B |
|-----------|----------------------------------|------------------|
| MIMIC CXR F1 | **88.9** | 81.1 |
| CheXpert F1 | **48.1** | 31.2 |
| SlakeVQA F1 | **62.3** | 38.6 |
| MedQA Accuracy | **87.7%** | 64.4% |
| MedMCQA Accuracy | **74.2%** | 55.7% |

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- **NVIDIA GPU** with CUDA support (recommended)
- **8GB+ RAM** (16GB+ recommended)
- **Hugging Face Account** with MedGemma model access

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/rahul188/MediVision-AI.git
cd MediVision-AI
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install PyTorch with CUDA** (for GPU acceleration)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

5. **Run the application**
```bash
streamlit run app.py
```

## 🔧 System Requirements

### Minimum Requirements
- **CPU**: Intel i5 or AMD Ryzen 5
- **RAM**: 8GB
- **Storage**: 10GB free space
- **GPU**: Optional (CPU mode available)

### Recommended Requirements
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9
- **RAM**: 16GB+
- **Storage**: 20GB+ SSD
- **GPU**: NVIDIA RTX 3060+ with 12GB+ VRAM

## 🖥️ GPU Support

MediVision-AI automatically detects and utilizes GPU acceleration:

- ✅ **NVIDIA RTX 4060 Ti** (tested)
- ✅ **CUDA 11.8+** support
- ✅ **Automatic device detection**
- ✅ **Memory optimization**

### Check GPU Status
The application displays real-time GPU information:
- GPU model and memory
- CUDA version
- PyTorch compatibility
- Memory usage monitoring

## 📚 Usage

### 1. Medical Image Analysis
- Upload medical images (PNG, JPG, JPEG)
- Select analysis type (General, Radiologist Review, Detailed Findings)
- Get AI-powered medical insights

### 2. Medical Q&A
- Ask medical questions in natural language
- Get evidence-based, professional responses
- Access predefined medical question templates

### 3. Report Generation
- Generate professional medical reports
- Support for radiology, pathology, and clinical summaries
- Download reports in text format

### 4. Performance Monitoring
- Real-time system monitoring
- GPU utilization tracking
- Performance metrics and logging

## 🔒 Important Disclaimers

⚠️ **FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

- This tool is a **proof of concept** demonstration
- **NOT for clinical diagnosis** or treatment decisions
- Always consult **qualified healthcare professionals**
- Outputs should be **reviewed by medical experts**

## 🏗️ Architecture

```
MediVision-AI/
├── app.py                 # Main Streamlit application
├── logger_utils.py        # Performance logging utilities
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── .streamlit/           # Streamlit configuration
│   └── config.toml       # App configuration
└── README.md             # This file
```

## 🔄 Development

### Running in Development Mode
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
streamlit run app.py --server.runOnSave true
```

### VS Code Tasks
Use the predefined VS Code tasks:
- **Setup MedGemma Environment**: Initialize the project
- **Start Streamlit App**: Launch the application
- **Check GPU Status**: Verify GPU configuration
- **Update Dependencies**: Install/update packages

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Research** for the MedGemma 4B-IT model
- **Hugging Face** for the transformers library
- **Streamlit** for the web framework
- **PyTorch** for deep learning capabilities

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/rahul188/MediVision-AI/issues)
- 📧 **Contact**: [Your Email]
- 📖 **Documentation**: [Wiki](https://github.com/rahul188/MediVision-AI/wiki)

## 🔗 Links

- [MedGemma Model](https://huggingface.co/google/medgemma-4b-it)
- [Streamlit Documentation](https://docs.streamlit.io)
- [PyTorch CUDA Setup](https://pytorch.org/get-started/locally/)

---

**Made with ❤️ for the medical AI community**

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
