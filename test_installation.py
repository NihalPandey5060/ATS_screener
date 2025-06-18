#!/usr/bin/env python3
"""
Test script to verify installation of all dependencies
Run this script before starting the main application
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name} - FAILED: {e}")
        return False

def main():
    """Test all required dependencies"""
    print("🔍 Testing AI Resume Screening System Dependencies")
    print("=" * 50)
    
    # List of required packages
    packages = [
        ("streamlit", "Streamlit"),
        ("fitz", "PyMuPDF"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sentence_transformers", "Sentence Transformers"),
        ("sklearn", "Scikit-learn"),
        ("plotly", "Plotly"),
    ]
    
    all_passed = True
    
    for module, package in packages:
        if not test_import(module, package):
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 All dependencies are installed correctly!")
        print("🚀 You can now run: streamlit run resume_screener.py")
    else:
        print("❌ Some dependencies are missing.")
        print("📦 Please run: pip install -r requirements.txt")
    
    # Test sentence transformer model
    print("\n🧠 Testing Sentence Transformer Model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence Transformer model loaded successfully!")
        
        # Test embedding generation
        test_text = "This is a test sentence."
        embedding = model.encode([test_text])
        print(f"✅ Embedding generation works! Shape: {embedding.shape}")
        
    except Exception as e:
        print(f"❌ Sentence Transformer test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎯 System is ready to use!")
    else:
        print("⚠️  Please fix the issues above before running the application.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 