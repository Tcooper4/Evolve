# Evolve Clean

## Overview
Evolve Clean is a comprehensive automation and trading dashboard project designed to streamline and enhance trading operations through automation, monitoring, and analytics. This project integrates various components including automation services, a web dashboard, monitoring systems, and extensive documentation.

## Project Structure

### Root Directory
- **.gitignore**: Configuration file to exclude sensitive and unnecessary files from version control.
- **.env**: Environment variables file (not tracked in git).
- **docker-compose.yml**: Docker Compose configuration for orchestrating services.
- **Dockerfile**: Base Dockerfile for building the application.
- **config.example.json**: Example configuration file for the application.
- **requirements.txt**: Python dependencies for the project.
- **setup.py**: Setup script for the project.
- **issues.md**: Document tracking project issues and tasks.

### Automation
The `automation` directory contains the core automation logic and services.

#### Core Components
- **core/**: Contains core automation logic including security, integration testing, logging, UI handling, performance handling, RBAC, service management, template engine, step handlers, workflow engine, task manager, orchestrator, and task handlers.
- **services/**: Contains various automation services including task management, health checks, metrics, security, logging, configuration, CLI, API, scheduler, notification, monitoring, workflows, tasks, and core services.
- **agents/**: Contains agents for documentation, analytics, versioning, management, orchestration, data collection, and more.
- **models/**: Contains automation models.
- **notifications/**: Contains notification services, cleanup, and handlers for webhooks, Slack, and email.
- **web/**: Contains web application components including dashboard, app, routes, templates, and static files.
- **api/**: Contains API endpoints for metrics and tasks.
- **monitoring/**: Contains alert management and metrics collection.
- **config/**: Contains deployment, test, notification, and backup configurations.
- **scripts/**: Contains deployment, monitoring setup, service deployment, secret management, backup system, environment setup, and pipeline scripts.
- **docs/**: Contains comprehensive documentation including contributing guidelines, versioning, support, changelog, troubleshooting, testing, deployment, security, performance, UI, integration testing, notification, data processing, logging, monitoring, RBAC, service management, and API documentation.
- **templates/**: Contains templates for notifications, API, system, code, HTML export, dashboard, webhook, Slack, and email.
- **tests/**: Contains integration, performance, chaos, stress, and various test suites for automation services, notifications, web app, metrics API, task API, alert manager, agents, orchestrator, data processing, model training, and documentation.
- **grafana/**: Contains Grafana dashboards, datasources, and provisioning configurations.
- **prometheus/**: Contains Prometheus configuration and alert rules.

### Dashboard
The `dashboard` directory contains the web dashboard components.

- **App.js**: Main application component.
- **index.js**: Entry point for the dashboard.
- **pages/**: Contains various pages for the dashboard.
- **components/**: Contains reusable components for the dashboard.
- **utils/**: Contains utility functions for the dashboard.
- **logs/**: Contains log files for the dashboard.

### Scripts
The `scripts` directory contains deployment scripts for Kubernetes and local environments.

- **deploy_kubernetes.sh**: Script for deploying to Kubernetes.
- **deploy_local.sh**: Script for deploying locally.

### Kubernetes
The `kubernetes` directory contains Kubernetes deployment configurations.

- **deployment.yaml**: Kubernetes deployment configuration.

### Rules
The `rules` directory contains alert rules.

- **alerts.yml**: Alert rules configuration.

### Alerts
The `alerts` directory contains alert management components.

- **README.md**: Documentation for alerts.
- **test_alert_manager.py**: Tests for the alert manager.
- **alert_manager.py**: Alert management logic.

### Config
The `config` directory contains configuration files.

- **config.json**: Main configuration file.

### Tests
The `tests` directory contains various test suites.

- **benchmark/**: Contains benchmark tests.
- **integration/**: Contains integration tests.
- **unit/**: Contains unit tests.
- **implementation/**: Contains implementation tests.

### Trading
The `trading` directory contains trading-related components.

- **models/**: Contains trading models.
- **web/**: Contains web components for trading.
- **visualization/**: Contains visualization components for trading.
- **risk/**: Contains risk management components.
- **portfolio/**: Contains portfolio management components.
- **utils/**: Contains utility functions for trading.
- **nlp/**: Contains natural language processing components for trading.
- **execution/**: Contains execution components for trading.
- **evaluation/**: Contains evaluation components for trading.
- **data/**: Contains data components for trading.
- **config/**: Contains configuration for trading.
- **backtesting/**: Contains backtesting components for trading.
- **strategies/**: Contains trading strategies.
- **optimizers/**: Contains optimization components for trading.
- **llm/**: Contains language model components for trading.
- **feature_engineering/**: Contains feature engineering components for trading.
- **analysis/**: Contains analysis components for trading.
- **agents/**: Contains agents for trading.
- **admin/**: Contains admin components for trading.
- **optimization/**: Contains optimization components for trading.
- **visuals/**: Contains visual components for trading.
- **cache/**: Contains cache components for trading.
- **logs/**: Contains log files for trading.

### Mock Data
The `mock_data` directory contains mock data for testing and development.

- **portfolio/**: Contains mock portfolio data.
- **market/**: Contains mock market data.
- **integration/**: Contains mock integration data.
- **implementation/**: Contains mock implementation data.
- **backtest/**: Contains mock backtest data.
- **alerts/**: Contains mock alert data.
- **db/**: Contains mock database data.
- **api/**: Contains mock API data.
- **indicators/**: Contains mock indicator data.
- **ml/**: Contains mock machine learning data.

### Models
The `models` directory contains model files.

### Test Model Save
The `test_model_save` directory contains saved model files.

- **tcn_model.pt**: Saved TCN model.

### Trading Dashboard Egg Info
The `trading_dashboard.egg-info` directory contains egg info for the trading dashboard.

- **SOURCES.txt**: Source files for the trading dashboard.
- **top_level.txt**: Top-level modules for the trading dashboard.
- **requires.txt**: Required dependencies for the trading dashboard.
- **dependency_links.txt**: Dependency links for the trading dashboard.
- **PKG-INFO**: Package information for the trading dashboard.

## Getting Started
To get started with Evolve Clean, follow these steps:

1. **Clone the Repository**: Clone the repository to your local machine.
2. **Install Dependencies**: Install the required dependencies using `pip install -r requirements.txt`.
3. **Set Up Environment Variables**: Copy `config.example.json` to `config.json` and update the environment variables in `.env`.
4. **Run the Application**: Use Docker Compose to run the application with `docker-compose up`.

## Contributing
Please read the [contributing guidelines](automation/docs/contributing.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Thanks to all contributors who have helped in the development of this project.