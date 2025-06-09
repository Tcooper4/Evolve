# Automation System

A comprehensive automation system for managing and executing tasks, with real-time monitoring and a modern web interface.

## Features

- **Task Management**
  - Create, update, and delete tasks
  - Task dependency management
  - Task execution and monitoring
  - Task metrics and history

- **Agent System**
  - Distributed task execution
  - Agent health monitoring
  - Automatic task distribution
  - Agent metrics collection

- **Monitoring Dashboard**
  - Real-time system metrics
  - Task status visualization
  - Resource usage tracking
  - Performance analytics

- **Web Interface**
  - Modern, responsive design
  - Real-time updates
  - Interactive task management
  - System monitoring

## Architecture

The system is built with a modular architecture:

- **Core Components**
  - Orchestrator: Manages task execution and coordination
  - Task Handlers: Process different types of tasks
  - Agent System: Handles distributed execution
  - Metrics Collector: Gathers system and task metrics

- **Web Components**
  - Flask-based API server
  - WebSocket support for real-time updates
  - Modern frontend with Bootstrap and Chart.js
  - Responsive monitoring dashboard

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/automation-system.git
   cd automation-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the system:
   - Copy `config.example.json` to `config.json`
   - Update the configuration with your settings

## Usage

1. Start the system:
   ```bash
   python -m automation.web.app
   ```

2. Access the web interface:
   - Task Management: http://localhost:5000
   - Monitoring Dashboard: http://localhost:5000/dashboard

3. API Endpoints:
   - Tasks: `/api/tasks`
   - Metrics: `/api/metrics`
   - Agents: `/api/agents`

## Development

### Project Structure
```
automation/
├── api/            # API endpoints
├── core/           # Core system components
├── agents/         # Agent implementations
├── monitoring/     # Monitoring components
├── web/           # Web interface
│   ├── templates/ # HTML templates
│   └── static/    # Static assets
└── tests/         # Test suite
```

### Running Tests
```bash
pytest automation/tests/
```

### Code Style
The project uses:
- Black for code formatting
- Flake8 for linting
- MyPy for type checking
- isort for import sorting

Run the style checks:
```bash
black automation/
flake8 automation/
mypy automation/
isort automation/
```

## Configuration

The system is configured through `config.json`:

```json
{
  "redis": {
    "host": "localhost",
    "port": 6379,
    "db": 0
  },
  "ray": {
    "address": "auto",
    "namespace": "automation"
  },
  "kubernetes": {
    "in_cluster": false,
    "namespace": "automation"
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "automation/logs/orchestrator.log"
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask for the web framework
- Redis for task queue management
- Ray for distributed computing
- Chart.js for data visualization
- Bootstrap for the UI framework