# Production Deployment Script for Intelligent Trading System (Windows)
# Usage: .\scripts\deploy.ps1 [command]

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("deploy", "stop", "start", "restart", "health", "logs", "rebuild")]
    [string]$Command = "deploy",
    
    [Parameter(Mandatory=$false)]
    [string]$Service = ""
)

$ErrorActionPreference = "Stop"

# Configuration
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ComposeFile = Join-Path $ProjectRoot "docker-compose.yml"
$ConfigFile = Join-Path $ProjectRoot "config.prod.yaml"

Write-Host "=== Intelligent Trading System - Production Deployment ===" -ForegroundColor Cyan

# Functions
function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

# Check prerequisites
function Test-Prerequisites {
    Write-Host "Checking prerequisites..."
    
    # Check Docker
    try {
        $null = docker --version
        Write-Success "Docker is installed"
    } catch {
        Write-Error "Docker is not installed. Please install Docker Desktop first."
        exit 1
    }
    
    # Check Docker Compose
    try {
        $null = docker-compose --version
        Write-Success "Docker Compose is installed"
    } catch {
        Write-Error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    }
    
    # Check if config file exists
    if (-not (Test-Path $ConfigFile)) {
        Write-Error "Production config file not found: $ConfigFile"
        exit 1
    }
    Write-Success "Production config file found"
    
    # Check environment variables
    if (-not $env:BINANCE_API_KEY) {
        Write-Warning "BINANCE_API_KEY environment variable not set"
    }
    if (-not $env:BINANCE_API_SECRET) {
        Write-Warning "BINANCE_API_SECRET environment variable not set"
    }
}

# Setup directories
function Initialize-Directories {
    Write-Host "Setting up directories..."
    
    $directories = @(
        (Join-Path $ProjectRoot "data"),
        (Join-Path $ProjectRoot "logs"),
        (Join-Path $ProjectRoot "cache"),
        (Join-Path $ProjectRoot "monitoring\prometheus"),
        (Join-Path $ProjectRoot "monitoring\grafana\dashboards"),
        (Join-Path $ProjectRoot "monitoring\grafana\datasources")
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Success "Directories created"
}

# Build Docker images
function Build-Images {
    Write-Host "Building Docker images..."
    
    Push-Location $ProjectRoot
    try {
        docker-compose build
        Write-Success "Docker images built"
    } finally {
        Pop-Location
    }
}

# Start services
function Start-Services {
    Write-Host "Starting services..."
    
    Push-Location $ProjectRoot
    try {
        docker-compose up -d
        Write-Success "Services started"
    } finally {
        Pop-Location
    }
}

# Stop services
function Stop-Services {
    Write-Host "Stopping services..."
    
    Push-Location $ProjectRoot
    try {
        docker-compose down
        Write-Success "Services stopped"
    } finally {
        Pop-Location
    }
}

# Restart services
function Restart-Services {
    Write-Host "Restarting services..."
    
    Push-Location $ProjectRoot
    try {
        docker-compose restart
        Write-Success "Services restarted"
    } finally {
        Pop-Location
    }
}

# Check service health
function Show-Health {
    Write-Host "Checking service health..."
    
    Push-Location $ProjectRoot
    try {
        docker-compose ps
        
        Write-Host ""
        Write-Host "Service URLs:" -ForegroundColor Cyan
        Write-Host "  - Web Dashboard: http://localhost:5000"
        Write-Host "  - Grafana: http://localhost:3000 (admin/admin)"
        Write-Host "  - Prometheus: http://localhost:9090"
    } finally {
        Pop-Location
    }
}

# View logs
function Show-Logs {
    param([string]$ServiceName = "")
    
    Push-Location $ProjectRoot
    try {
        if ([string]::IsNullOrEmpty($ServiceName)) {
            docker-compose logs -f
        } else {
            docker-compose logs -f $ServiceName
        }
    } finally {
        Pop-Location
    }
}

# Main deployment function
function Deploy-System {
    Write-Host "Starting deployment..."
    
    Test-Prerequisites
    Initialize-Directories
    Build-Images
    Start-Services
    
    Write-Host ""
    Write-Success "Deployment completed successfully!"
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Check service health: .\scripts\deploy.ps1 health"
    Write-Host "  2. View logs: .\scripts\deploy.ps1 logs"
    Write-Host "  3. Access dashboard: http://localhost:5000"
}

# Main script
switch ($Command) {
    "deploy" {
        Deploy-System
    }
    "stop" {
        Stop-Services
    }
    "start" {
        Start-Services
    }
    "restart" {
        Restart-Services
    }
    "health" {
        Show-Health
    }
    "logs" {
        Show-Logs -ServiceName $Service
    }
    "rebuild" {
        Stop-Services
        Build-Images
        Start-Services
    }
    default {
        Write-Host "Usage: .\scripts\deploy.ps1 [command]" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor Cyan
        Write-Host "  deploy    - Deploy the system (default)"
        Write-Host "  stop      - Stop all services"
        Write-Host "  start     - Start all services"
        Write-Host "  restart   - Restart all services"
        Write-Host "  health    - Check service health"
        Write-Host "  logs      - View logs (optional: -Service name)"
        Write-Host "  rebuild   - Rebuild and restart services"
        exit 1
    }
}
