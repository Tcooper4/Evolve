#!/bin/bash

# ============================================================================
# EVOLVE TRADING PLATFORM - PRODUCTION DEPLOYMENT SCRIPT
# ============================================================================
# Comprehensive deployment script with security and health checks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/env.example"
DOCKER_COMPOSE_FILE="$SCRIPT_DIR/docker-compose.production.yml"

# Logging
LOG_FILE="$PROJECT_ROOT/logs/deployment.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f "$ENV_FILE" ]; then
        warning ".env file not found. Creating from template..."
        if [ -f "$ENV_EXAMPLE" ]; then
            cp "$ENV_EXAMPLE" "$ENV_FILE"
            warning "Please edit $ENV_FILE with your actual configuration values."
            exit 1
        else
            error "env.example file not found. Cannot create .env file."
            exit 1
        fi
    fi
    
    success "Prerequisites check passed"
}

# Function to validate environment variables
validate_environment() {
    log "Validating environment variables..."
    
    # Required variables
    required_vars=(
        "OPENAI_API_KEY"
        "FINNHUB_API_KEY"
        "ALPHA_VANTAGE_API_KEY"
        "APP_SECRET_KEY"
        "JWT_SECRET_KEY"
    )
    
    missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        error "Missing required environment variables: ${missing_vars[*]}"
        error "Please set these variables in your .env file."
        exit 1
    fi
    
    success "Environment validation passed"
}

# Function to run security checks
security_checks() {
    log "Running security checks..."
    
    # Check for default passwords
    if grep -q "your-secret-key-here" "$ENV_FILE"; then
        error "Default secret key detected. Please change APP_SECRET_KEY in .env file."
        exit 1
    fi
    
    if grep -q "your-jwt-secret-key-here" "$ENV_FILE"; then
        error "Default JWT secret detected. Please change JWT_SECRET_KEY in .env file."
        exit 1
    fi
    
    # Check for default API keys
    if grep -q "your-openai-api-key-here" "$ENV_FILE"; then
        error "Default OpenAI API key detected. Please set OPENAI_API_KEY in .env file."
        exit 1
    fi
    
    success "Security checks passed"
}

# Function to build and deploy
deploy() {
    log "Starting deployment..."
    
    # Load environment variables
    set -a
    source "$ENV_FILE"
    set +a
    
    # Validate environment
    validate_environment
    
    # Run security checks
    security_checks
    
    # Create necessary directories
    log "Creating necessary directories..."
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/models"
    mkdir -p "$PROJECT_ROOT/reports"
    mkdir -p "$PROJECT_ROOT/backtest_results"
    mkdir -p "$PROJECT_ROOT/backups"
    
    # Build and start services
    log "Building and starting services..."
    cd "$SCRIPT_DIR"
    
    # Pull latest images
    docker-compose -f "$DOCKER_COMPOSE_FILE" pull
    
    # Build application
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache evolve-trading
    
    # Start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    success "Deployment completed"
}

# Function to check deployment health
health_check() {
    log "Performing health checks..."
    
    # Wait for services to start
    log "Waiting for services to start..."
    sleep 30
    
    # Check application health
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        success "Application is healthy"
    else
        error "Application health check failed"
        return 1
    fi
    
    # Check database health
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_isready -U "${POSTGRES_USER:-evolve_user}" -d evolve_trading > /dev/null 2>&1; then
        success "Database is healthy"
    else
        error "Database health check failed"
        return 1
    fi
    
    # Check Redis health
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null 2>&1; then
        success "Redis is healthy"
    else
        error "Redis health check failed"
        return 1
    fi
    
    success "All health checks passed"
}

# Function to show deployment status
status() {
    log "Checking deployment status..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    echo ""
    log "Service URLs:"
    echo "  Application: http://localhost:8501"
    echo "  Grafana: http://localhost:3000"
    echo "  Prometheus: http://localhost:9090"
}

# Function to stop services
stop() {
    log "Stopping services..."
    cd "$SCRIPT_DIR"
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    success "Services stopped"
}

# Function to restart services
restart() {
    log "Restarting services..."
    stop
    deploy
    health_check
}

# Function to view logs
logs() {
    log "Showing logs..."
    cd "$SCRIPT_DIR"
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
}

# Function to backup data
backup() {
    log "Creating backup..."
    
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_dir="$PROJECT_ROOT/backups/backup_$timestamp"
    
    mkdir -p "$backup_dir"
    
    # Backup PostgreSQL
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T postgres pg_dump -U "${POSTGRES_USER:-evolve_user}" evolve_trading > "$backup_dir/postgres_backup.sql"
    
    # Backup MongoDB
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T mongodb mongodump --username "${MONGODB_USERNAME:-evolve_user}" --password "${MONGODB_PASSWORD:-evolve_password}" --db evolve_trading --out "$backup_dir/mongodb_backup"
    
    # Backup application data
    tar -czf "$backup_dir/app_data.tar.gz" -C "$PROJECT_ROOT" logs models reports backtest_results
    
    success "Backup created: $backup_dir"
}

# Function to show help
show_help() {
    echo "Evolve Trading Platform - Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy     - Deploy the application"
    echo "  status     - Show deployment status"
    echo "  health     - Perform health checks"
    echo "  stop       - Stop all services"
    echo "  restart    - Restart all services"
    echo "  logs       - Show service logs"
    echo "  backup     - Create backup of data"
    echo "  help       - Show this help message"
    echo ""
}

# Main script logic
main() {
    case "${1:-deploy}" in
        deploy)
            check_prerequisites
            deploy
            health_check
            status
            ;;
        status)
            status
            ;;
        health)
            health_check
            ;;
        stop)
            stop
            ;;
        restart)
            restart
            ;;
        logs)
            logs
            ;;
        backup)
            backup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 