#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${YELLOW}[*] $1${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}[+] $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[-] $1${NC}"
}

# Create necessary directories
print_status "Creating test directories..."
mkdir -p test_results
mkdir -p coverage_reports

# Install test dependencies
print_status "Installing test dependencies..."
pip install -r requirements-test.txt

# Run tests with coverage
print_status "Running tests with coverage..."
pytest tests/ \
    --cov=trading \
    --cov-report=term-missing \
    --cov-report=html:coverage_reports/html \
    --junitxml=test_results/junit.xml \
    --html=test_results/report.html \
    -v

# Check if tests passed
if [ $? -eq 0 ]; then
    print_success "All tests passed!"
else
    print_error "Some tests failed. Check test_results/ for details."
    exit 1
fi

# Generate coverage report
print_status "Generating coverage report..."
coverage html -d coverage_reports/html

# Print summary
print_status "Test Results Summary:"
echo "----------------------"
echo "HTML Report: test_results/report.html"
echo "JUnit Report: test_results/junit.xml"
echo "Coverage Report: coverage_reports/html/index.html"

print_success "Test execution completed!" 