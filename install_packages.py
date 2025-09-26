import subprocess
import sys

# List of required packages
packages = [
    "streamlit",
    "langchain",
    "langchain-google-genai",
    "google-genai",
    "faiss-cpu",
    "sentence-transformers",
    "python-dotenv",
    "langchain-community",
    "langchain-huggingface",
    "instructor-embedding"
]

for package in packages:
    try:
        __import__(package.replace("-", "_"))  # replace '-' with '_' for import
    except ImportError:
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{package} is already installed.")
