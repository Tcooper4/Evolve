import json
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_directory_structure():
    """Create the necessary directory structure."""
    directories = [
        "automation/agents",
        "automation/config",
        "automation/scripts",
        "automation/logs",
        "automation/results",
        "automation/tests",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def create_config_files():
    """Create default configuration files."""
    # Default config
    default_config = {
        "openai": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 2000},
        "agents": {
            "code_generation": {"enabled": True, "priority": 1},
            "testing": {"enabled": True, "priority": 2},
            "review": {"enabled": True, "priority": 3},
            "deployment": {"enabled": True, "priority": 4},
        },
        "paths": {"code_base": "trading", "tests": "tests", "docs": "docs"},
        "monitoring": {
            "enabled": True,
            "check_interval": 60,
            "alert_thresholds": {"cpu": 80, "memory": 85, "disk": 90},
        },
        "error_handling": {"max_retries": 3, "retry_delay": 5, "recovery_enabled": True},
    }

    # Save config
    config_path = "automation/config/config.json"
    with open(config_path, "w") as f:
        json.dump(default_config, f, indent=4)
    logger.info(f"Created config file: {config_path}")


def create_requirements_file():
    """Create requirements.txt file."""
    requirements = [
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
        "psutil>=5.8.0",
        "aiohttp>=3.8.0",
        "asyncio>=3.4.3",
        "pytest>=6.2.5",
        "pytest-asyncio>=0.16.0",
        "black>=21.9b0",
        "isort>=5.9.3",
        "mypy>=0.910",
        "pylint>=2.9.6",
    ]

    with open("automation/requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    logger.info("Created requirements.txt file")


def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = """

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
*.log
logs/

# Environment variables
.env

# Test coverage
.coverage
htmlcov/

# Distribution
dist/
build/

# Jupyter Notebook
.ipynb_checkpoints

# Local development
.DS_Store
"""

    with open("automation/.gitignore", "w") as f:
        f.write(gitignore_content.strip())
    logger.info("Created .gitignore file")


def create_readme():
    """Create README.md file."""
    readme_content = """# AI Development Automation

This project implements an automated development pipeline using AI agents to handle various development tasks.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

2. Install dependencies:
```bash
pip install -r automation/requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

## Usage

Run the automation pipeline:
```bash
python automation/scripts/run_pipeline.py --tasks automation/config/tasks.json --config automation/config/config.json
```

## Configuration

- `config.json`: Main configuration file
- `tasks.json`: Task definitions and requirements

## Directory Structure

- `agents/`: AI agent implementations
- `config/`: Configuration files
- `scripts/`: Utility scripts
- `logs/`: Log files
- `results/`: Task results and outputs
- `tests/`: Test files

## Monitoring

The system includes comprehensive monitoring:
- System metrics (CPU, memory, disk)
- API health checks
- Error tracking and recovery
- Performance metrics

## Error Handling

The system implements robust error handling:
- Automatic retries
- Error recovery strategies
- Detailed error logging
- Error statistics and reporting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
"""

    with open("automation/README.md", "w") as f:
        f.write(readme_content)
    logger.info("Created README.md file")


def create_env_example():
    """Create .env.example file."""
    env_content = """# OpenAI API Configuration

OPENAI_API_KEY=your_api_key_here

# System Configuration
LOG_LEVEL=INFO
MAX_RETRIES=3
RETRY_DELAY=5

# Monitoring Configuration
ENABLE_MONITORING=true
CHECK_INTERVAL=60
CPU_THRESHOLD=80
MEMORY_THRESHOLD=85
DISK_THRESHOLD=90
"""

    with open("automation/.env.example", "w") as f:
        f.write(env_content)
    logger.info("Created .env.example file")


def main():
    """Main setup function."""
    try:
        # Create directory structure
        create_directory_structure()

        # Create configuration files
        create_config_files()

        # Create requirements file
        create_requirements_file()

        # Create .gitignore
        create_gitignore()

        # Create README
        create_readme()

        # Create .env.example
        create_env_example()

        logger.info("Environment setup completed successfully")

    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
