#!/usr/bin/env python3
"""
Dependency Resolution Script

This script helps resolve pip-compile issues with problematic packages
like dowhy and empyrical by testing different configurations and
providing solutions.
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True, 
            check=False
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_pip_compile():
    """Test pip-compile with the current requirements.in."""
    print("ðŸ” Testing pip-compile with current requirements.in...")
    
    success, stdout, stderr = run_command("pip-compile requirements.in --upgrade")
    
    if success:
        print("âœ… pip-compile succeeded!")
        return True
    else:
        print("âŒ pip-compile failed:")
        print(stderr)
        return False

def test_problematic_packages():
    """Test installation of problematic packages individually."""
    problematic_packages = [
        "dowhy==0.8",
        "empyrical==0.5.5",
        "ray==2.8.0",
        "bentoml==1.4.17"
    ]
    
    print("\nðŸ” Testing problematic packages individually...")
    
    for package in problematic_packages:
        print(f"\nTesting {package}...")
        success, stdout, stderr = run_command(f"pip install {package}")
        
        if success:
            print(f"âœ… {package} installed successfully")
        else:
            print(f"âŒ {package} failed to install:")
            print(stderr[:500] + "..." if len(stderr) > 500 else stderr)

def create_alternative_requirements():
    """Create alternative requirements files for different scenarios."""
    
    # Minimal requirements for core functionality
    minimal_requirements = """# Minimal requirements for core trading functionality
numpy>=2.0.0,<3.0.0
pandas>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0
scipy>=1.10.0,<2.0.0
matplotlib>=3.6.0,<4.0.0
torch>=2.0.0,<3.0.0
xgboost>=1.7.0,<2.0.0
fastapi>=0.100.0,<1.0.0
streamlit>=1.25.0,<2.0.0
redis>=4.5.0,<5.0.0
python-dotenv>=1.0.0,<2.0.0
pytest>=7.4.0,<8.0.0
"""
    
    # Requirements without problematic packages
    stable_requirements = """# Stable requirements without problematic packages
numpy>=2.0.0,<3.0.0
pandas>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0
scipy>=1.10.0,<2.0.0
matplotlib>=3.6.0,<4.0.0
seaborn>=0.12.0,<1.0.0
torch>=2.0.0,<3.0.0
transformers>=4.30.0,<5.0.0
xgboost>=1.7.0,<2.0.0
lightgbm>=4.0.0,<5.0.0
prophet>=1.1.0,<2.0.0
statsmodels>=0.14.0,<1.0.0
optuna>=3.0.0,<4.0.0
shap>=0.42.0,<1.0.0
fastapi>=0.100.0,<1.0.0
streamlit>=1.25.0,<2.0.0
redis>=4.5.0,<5.0.0
sqlalchemy>=2.0.0,<3.0.0
python-dotenv>=1.0.0,<2.0.0
pydantic>=2.0.0,<3.0.0
aiohttp>=3.8.0,<4.0.0
cryptography>=41.0.0,<42.0.0
nltk>=3.8.0,<4.0.0
plotly>=5.15.0,<6.0.0
pytest>=7.4.0,<8.0.0
black>=23.7.0,<24.0.0
flake8>=6.0.0,<7.0.0
"""
    
    # Write alternative requirements files
    with open("requirements-minimal.in", "w") as f:
        f.write(minimal_requirements)
    
    with open("requirements-stable.in", "w") as f:
        f.write(stable_requirements)
    
    print("ðŸ“ Created alternative requirements files:")
    print("  - requirements-minimal.in (core functionality only)")
    print("  - requirements-stable.in (stable packages only)")

def test_alternative_requirements():
    """Test pip-compile with alternative requirements files."""
    print("\nðŸ” Testing alternative requirements files...")
    
    for req_file in ["requirements-minimal.in", "requirements-stable.in"]:
        if os.path.exists(req_file):
            print(f"\nTesting {req_file}...")
            success, stdout, stderr = run_command(f"pip-compile {req_file} --upgrade")
            
            if success:
                print(f"âœ… {req_file} compiled successfully")
            else:
                print(f"âŒ {req_file} failed:")
                print(stderr[:300] + "..." if len(stderr) > 300 else stderr)

def generate_recommendations():
    """Generate recommendations for dependency management."""
    print("\nðŸ“‹ Dependency Management Recommendations:")
    print("\n1. Use requirements-stable.in for production:")
    print("   pip-compile requirements-stable.in --upgrade --generate-hashes")
    
    print("\n2. For development, use requirements-dev.txt separately:")
    print("   pip install -r requirements-stable.txt")
    print("   pip install -r requirements-dev.txt")
    
    print("\n3. Problematic packages to install separately if needed:")
    print("   - dowhy: pip install dowhy==0.8")
    print("   - empyrical: pip install empyrical==0.5.5")
    print("   - ray: pip install ray==2.8.0")
    print("   - bentoml: pip install bentoml==1.4.17")
    
    print("\n4. Use virtual environments for different configurations:")
    print("   python -m venv venv-stable")
    print("   python -m venv venv-full")
    
    print("\n5. Consider using conda for complex scientific packages:")
    print("   conda install -c conda-forge dowhy empyrical")

def main():
    """Main function to run dependency resolution."""
    print("ðŸš€ Dependency Resolution Script")
    print("=" * 50)
    
    # Test current setup
    if not test_pip_compile():
        print("\nâš ï¸  Current requirements.in has issues")
    
    # Test problematic packages
    test_problematic_packages()
    
    # Create alternative requirements
    create_alternative_requirements()
    
    # Test alternatives
    test_alternative_requirements()
    
    # Generate recommendations
    generate_recommendations()
    
    print("\nâœ… Dependency resolution analysis complete!")

if __name__ == "__main__":
    main()
