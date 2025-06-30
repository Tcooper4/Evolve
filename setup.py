from setuptools import setup, find_packages
from typing import List

def read_file(filename: str) -> str:
    """Read file contents as string."""
    with open(filename, "r", encoding="utf-8") as fh:
        return {'success': True, 'result': fh.read(), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def read_requirements(filename: str) -> List[str]:
    """Read requirements from file, filtering out comments and empty lines."""
    with open(filename, "r", encoding="utf-8") as fh:
        return {'success': True, 'result': [line.strip() for line in fh if line.strip() and not line.startswith("#")], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

# Read project files
long_description = read_file("README.md")
requirements = read_requirements("requirements.txt")

setup(
    name="evolve-trading",
    version="1.0.0",
    author="Thomas Cooper",
    author_email="thomas.cooper@example.com",
    description="Autonomous financial forecasting and trading strategy platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tcooper4/Evolve",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.15.0",
            "pytest-mock>=3.6.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.900",
            "flake8>=4.0.0",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "mkdocs>=1.2.0",
            "mkdocs-material>=7.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "evolve=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "trading": [
            "config/*.yaml",
            "config/*.json",
            "data/*.csv",
            "models/*.pkl",
        ],
    },
) 