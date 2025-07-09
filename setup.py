#!/usr/bin/env python3
"""
Setup script for MedGemma 4B-IT POC
This script helps setup the environment and authenticate with Hugging Face
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name} ({gpu_count} device(s))")
            return True
        else:
            print("⚠️  No CUDA-compatible GPU detected. Will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet. GPU check will be performed after installation.")
        return False

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    if success:
        print("✅ Requirements installed successfully")
        return True
    else:
        print(f"❌ Error installing requirements: {stderr}")
        return False

def setup_huggingface():
    """Setup Hugging Face authentication"""
    print("\n🤗 Setting up Hugging Face authentication...")
    print("You need to:")
    print("1. Create an account at https://huggingface.co")
    print("2. Visit https://huggingface.co/google/medgemma-4b-it")
    print("3. Accept the Health AI Developer Foundation's terms of use")
    print("4. Get your access token from https://huggingface.co/settings/tokens")
    
    # Check if already logged in
    success, stdout, stderr = run_command("huggingface-cli whoami", check=False)
    if success:
        print(f"✅ Already logged in to Hugging Face as: {stdout.strip()}")
        return True
    
    print("\n🔑 Please login to Hugging Face:")
    success, stdout, stderr = run_command("huggingface-cli login")
    
    if success:
        print("✅ Hugging Face authentication successful")
        return True
    else:
        print("❌ Hugging Face authentication failed")
        print("Please run 'huggingface-cli login' manually")
        return False

def create_sample_config():
    """Create a sample configuration file"""
    config_content = """# MedGemma POC Configuration
# You can modify these settings as needed

[model]
model_id = "google/medgemma-4b-it"
torch_dtype = "bfloat16"  # or "float32" for CPU
device_map = "auto"       # or "cpu" to force CPU usage
max_new_tokens = 300

[app]
page_title = "MedGemma 4B-IT POC"
page_icon = "🏥"
layout = "wide"

[performance]
# Set to false if you have limited GPU memory
use_gpu = true
# Set to true for slower but more memory-efficient loading
low_memory_mode = false
"""
    
    with open("config.ini", "w") as f:
        f.write(config_content)
    print("✅ Created sample configuration file: config.ini")

def main():
    """Main setup function"""
    print("🚀 MedGemma 4B-IT POC Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check GPU after installation
    check_gpu()
    
    # Setup Hugging Face
    if not setup_huggingface():
        print("⚠️  You can setup Hugging Face authentication later")
    
    # Create sample config
    create_sample_config()
    
    print("\n" + "=" * 40)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: streamlit run app.py")
    print("2. Open your browser to the displayed URL")
    print("3. Load the MedGemma model using the sidebar")
    print("4. Start exploring medical AI capabilities!")
    
    print("\n⚠️  Remember:")
    print("- First model load will download ~9GB")
    print("- Ensure you have accepted MedGemma terms on Hugging Face")
    print("- This is for research/educational use only")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
