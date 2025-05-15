#!/bin/bash

# Script to set up and run the Loan Approval ETL Pipeline

# Function to display help message
show_help() {
    echo "Loan Approval ETL Pipeline"
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start       Start the Airflow and MLflow services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  logs        Show logs from all services"
    echo "  status      Check the status of all services"
    echo "  help        Show this help message"
    echo ""
}

# Function to start services
start_services() {
    echo "Starting Airflow and MLflow services..."
    docker-compose up -d
    echo "Services started. Access Airflow UI at http://localhost:8080 and MLflow UI at http://localhost:5000"
}

# Function to stop services
stop_services() {
    echo "Stopping all services..."
    docker-compose down
    echo "All services stopped."
}

# Function to restart services
restart_services() {
    echo "Restarting all services..."
    docker-compose down
    docker-compose up -d
    echo "Services restarted. Access Airflow UI at http://localhost:8080 and MLflow UI at http://localhost:5000"
}

# Function to show logs
show_logs() {
    echo "Showing logs from all services..."
    docker-compose logs -f
}

# Function to check status
check_status() {
    echo "Checking status of all services..."
    docker-compose ps
}

# Main script logic
case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs
        ;;
    status)
        check_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

exit 0
