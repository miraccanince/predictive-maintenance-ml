#!/bin/bash

# =============================================================================
# SentinelX Test Runner
# =============================================================================
# Automated script to run Playwright E2E tests
#
# Usage:
#   ./run-tests.sh          # Run all tests
#   ./run-tests.sh api      # Run API tests only
#   ./run-tests.sh ui       # Run Dashboard UI tests only
#   ./run-tests.sh headed   # Run tests with visible browser
#   ./run-tests.sh setup    # Install dependencies only
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print with color
print_status() {
    echo -e "${BLUE}[SentinelX]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if services are healthy
check_services() {
    print_status "Checking services..."

    # Check API
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API is healthy (http://localhost:8000)"
    else
        print_warning "API is not responding. Starting services..."
        return 1
    fi

    # Check Dashboard
    if curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        print_success "Dashboard is healthy (http://localhost:8501)"
    else
        print_warning "Dashboard is not responding. Starting services..."
        return 1
    fi

    return 0
}

# Start services with docker-compose
start_services() {
    print_status "Starting SentinelX services..."
    docker-compose up -d --build

    print_status "Waiting for services to be ready..."

    # Wait for API
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_success "API is ready"
            break
        fi
        echo -n "."
        sleep 2
    done

    # Wait for Dashboard
    for i in {1..30}; do
        if curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
            print_success "Dashboard is ready"
            break
        fi
        echo -n "."
        sleep 2
    done

    echo ""
}

# Install dependencies
install_deps() {
    print_status "Installing Node.js dependencies..."

    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install Node.js first."
        exit 1
    fi

    npm install

    print_status "Installing Playwright browsers..."
    npx playwright install chromium

    print_success "Dependencies installed"
}

# Run tests
run_tests() {
    local test_type=$1

    case $test_type in
        "api")
            print_status "Running API tests..."
            npx playwright test tests/api.spec.ts --project=chromium
            ;;
        "ui"|"dashboard")
            print_status "Running Dashboard UI tests..."
            npx playwright test tests/dashboard.spec.ts --project=chromium
            ;;
        "headed")
            print_status "Running tests with visible browser..."
            npx playwright test --headed --project=chromium
            ;;
        "debug")
            print_status "Running tests in debug mode..."
            npx playwright test --debug --project=chromium
            ;;
        "report")
            print_status "Opening test report..."
            npx playwright show-report
            ;;
        *)
            print_status "Running all tests..."
            npx playwright test --project=chromium
            ;;
    esac
}

# Main script
main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║           SentinelX Playwright Test Runner                ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""

    cd "$(dirname "$0")"

    case "${1:-}" in
        "setup")
            check_docker
            install_deps
            print_success "Setup complete! Run './run-tests.sh' to execute tests."
            ;;
        "start")
            check_docker
            start_services
            ;;
        "stop")
            print_status "Stopping services..."
            docker-compose down
            print_success "Services stopped"
            ;;
        "report")
            run_tests "report"
            ;;
        *)
            check_docker

            # Check if dependencies are installed
            if [ ! -d "node_modules" ]; then
                print_warning "Dependencies not installed. Running setup..."
                install_deps
            fi

            # Check if services are running
            if ! check_services; then
                start_services
            fi

            # Run tests
            run_tests "$1"

            # Show summary
            echo ""
            print_success "Tests complete! Run './run-tests.sh report' to view the HTML report."
            ;;
    esac
}

main "$@"
