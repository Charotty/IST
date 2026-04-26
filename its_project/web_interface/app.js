// Intelligent Trading System - Web Interface JavaScript

const BASE_URL = 'http://127.0.0.1:5050';

// Global state
let currentTab = 'data';
let backendConnected = false;
let dataLoaded = false;
let featuresCalculated = false;
let modelTrained = false;
let backtestRun = false;
let modelDeployed = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeForms();
    checkBackendHealth();
    loadSavedModels();
    setInterval(checkBackendHealth, 5000);
});

// Tab Navigation
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            
            // Update active button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Update active content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tabName}-tab`) {
                    content.classList.add('active');
                }
            });
            
            currentTab = tabName;
        });
    });
}

// Form Initialization
function initializeForms() {
    // Data Loading
    document.getElementById('loadDataBtn').addEventListener('click', loadData);
    
    // Feature Engineering
    document.getElementById('calculateFeaturesBtn').addEventListener('click', calculateFeatures);
    
    // Model Training
    document.getElementById('trainModelBtn').addEventListener('click', trainModel);
    document.getElementById('saveModelBtn').addEventListener('click', saveModel);
    document.getElementById('loadModelBtn').addEventListener('click', loadSelectedModel);
    document.getElementById('trainTestSplit').addEventListener('input', (e) => {
        document.getElementById('trainTestSplitValue').textContent = `${Math.round(e.target.value * 100)}% train`;
    });
    
    // Backtesting
    document.getElementById('runBacktestBtn').addEventListener('click', runBacktest);
    
    // Deployment
    document.getElementById('deployModelBtn').addEventListener('click', deployModel);
    document.getElementById('stopDeploymentBtn').addEventListener('click', stopDeployment);
    
    // Set default dates
    const endDate = new Date();
    const startDate = new Date();
    startDate.setMonth(startDate.getMonth() - 3);
    
    document.getElementById('dataEndDate').value = endDate.toISOString().split('T')[0];
    document.getElementById('dataStartDate').value = startDate.toISOString().split('T')[0];
}

// Backend Health Check
async function checkBackendHealth() {
    try {
        const response = await fetch(`${BASE_URL}/api/health`);
        const data = await response.json();
        
        if (data.healthy) {
            backendConnected = true;
            updateBackendStatus(true);
        } else {
            backendConnected = false;
            updateBackendStatus(false);
        }
    } catch (error) {
        backendConnected = false;
        updateBackendStatus(false);
    }
}

function updateBackendStatus(connected) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    
    if (connected) {
        statusDot.classList.add('connected');
        statusDot.classList.remove('connecting');
        statusText.textContent = 'Backend: Connected';
    } else {
        statusDot.classList.remove('connected');
        statusText.textContent = 'Backend: Disconnected';
    }
}

// API Request Helper
async function apiRequest(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(`${BASE_URL}${endpoint}`, options);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.message || 'API request failed');
        }
        
        return result;
    } catch (error) {
        console.error('API request error:', error);
        return { success: false, message: error.message };
    }
}

// Data Loading
async function loadData() {
    const symbol = document.getElementById('dataSymbol').value;
    const timeframe = document.getElementById('dataTimeframe').value;
    const startDate = document.getElementById('dataStartDate').value;
    const endDate = document.getElementById('dataEndDate').value;
    
    const dataSources = [];
    document.querySelectorAll('.data-panel input[type="checkbox"]:checked').forEach(cb => {
        dataSources.push(cb.value);
    });
    
    if (!startDate || !endDate) {
        alert('Please select start and end dates');
        return;
    }
    
    if (dataSources.length === 0) {
        alert('Please select at least one data source');
        return;
    }
    
    const statusElement = document.getElementById('dataStatus');
    statusElement.querySelector('.value').textContent = 'Loading...';
    
    const result = await apiRequest('/api/data/load', 'POST', {
        symbol,
        timeframe,
        start_date: startDate,
        end_date: endDate,
        data_sources: dataSources
    });
    
    if (result.success) {
        dataLoaded = true;
        statusElement.querySelector('.value').textContent = `Data loaded: ${result.data.candle_count} candles`;
        document.getElementById('calculateFeaturesBtn').disabled = false;
    } else {
        statusElement.querySelector('.value').textContent = `Error: ${result.message}`;
    }
}

// Feature Engineering
async function calculateFeatures() {
    const technicalIndicators = [];
    document.querySelectorAll('.features-panel .checkbox-group:first-of-type input:checked').forEach(cb => {
        technicalIndicators.push(cb.value);
    });
    
    const orderbookFeatures = [];
    document.querySelectorAll('.features-panel .checkbox-group:last-of-type input:checked').forEach(cb => {
        orderbookFeatures.push(cb.value);
    });
    
    const statusElement = document.getElementById('featuresStatus');
    statusElement.querySelector('.value').textContent = 'Calculating...';
    
    const result = await apiRequest('/api/features/calculate', 'POST', {
        technical_indicators: technicalIndicators,
        orderbook_features: orderbookFeatures
    });
    
    if (result.success) {
        featuresCalculated = true;
        statusElement.querySelector('.value').textContent = `Features calculated: ${result.data.total_features} features`;
        document.getElementById('trainModelBtn').disabled = false;
    } else {
        statusElement.querySelector('.value').textContent = `Error: ${result.message}`;
    }
}

// Model Training
async function trainModel() {
    const modelType = document.getElementById('modelType').value;
    const sequenceLength = parseInt(document.getElementById('sequenceLength').value);
    const hiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
    const dropout = parseFloat(document.getElementById('dropout').value);
    const learningRate = parseFloat(document.getElementById('learningRate').value);
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    const trainTestSplit = parseFloat(document.getElementById('trainTestSplit').value);
    
    const statusElement = document.getElementById('trainingStatus');
    const progressElement = document.getElementById('trainingProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    statusElement.querySelector('.value').textContent = 'Training...';
    progressElement.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = 'Initializing...';
    
    // Start training in background
    const startResult = await apiRequest('/api/models/train', 'POST', {
        model_type: modelType,
        sequence_length: sequenceLength,
        hidden_layers: hiddenLayers,
        dropout: dropout,
        learning_rate: learningRate,
        epochs: epochs,
        batch_size: batchSize,
        train_test_split: trainTestSplit
    });
    
    if (!startResult.success) {
        statusElement.querySelector('.value').textContent = `Error: ${startResult.message}`;
        progressElement.style.display = 'none';
        return;
    }
    
    // Poll for progress until training completes
    const progressInterval = setInterval(async () => {
        try {
            const progressResult = await apiRequest('/api/models/training-progress', 'GET');
            if (progressResult.success && progressResult.data) {
                const progress = progressResult.data;
                const percentage = progress.percentage || 0;
                const currentEpoch = progress.current_epoch || 0;
                const totalEpochs = progress.total_epochs || 0;
                const currentLoss = progress.current_loss || 0;
                
                if (totalEpochs > 0) {
                    progressFill.style.width = `${percentage}%`;
                    progressText.textContent = `${Math.round(percentage)}% (Epoch ${currentEpoch}/${totalEpochs}, Loss: ${currentLoss.toFixed(4)})`;
                } else {
                    progressText.textContent = 'Initializing...';
                }
                
                // Check if training is complete
                if (!progress.is_training && currentEpoch > 0) {
                    clearInterval(progressInterval);
                    modelTrained = true;
                    statusElement.querySelector('.value').textContent = 'Model trained successfully';
                    progressFill.style.width = '100%';
                    progressText.textContent = '100%';
                    document.getElementById('saveModelBtn').disabled = false;
                    document.getElementById('runBacktestBtn').disabled = false;
                    document.getElementById('deployModelBtn').disabled = false;
                    
                    // Get final metrics from backend
                    const metricsResult = await apiRequest('/api/models/metrics', 'GET');
                    if (metricsResult.success && metricsResult.data) {
                        document.getElementById('metricAccuracy').textContent = metricsResult.data.accuracy.toFixed(3);
                        document.getElementById('metricPrecision').textContent = metricsResult.data.precision.toFixed(3);
                        document.getElementById('metricRecall').textContent = metricsResult.data.recall.toFixed(3);
                        document.getElementById('metricF1').textContent = metricsResult.data.f1.toFixed(3);
                    }
                }
            }
        } catch (error) {
            console.error('Error fetching progress:', error);
        }
    }, 500);
}

async function saveModel() {
    const modelName = prompt('Enter model name:', 'my_model');
    if (!modelName) return;
    
    const result = await apiRequest('/api/models/save', 'POST', {
        model_name: modelName
    });
    
    if (result.success) {
        alert(`Model saved as ${result.data.filename}`);
        loadSavedModels(); // Refresh model list
    } else {
        alert(`Error saving model: ${result.message}`);
    }
}

async function loadSavedModels() {
    const result = await apiRequest('/api/models/list', 'GET');
    if (result.success) {
        // Update Deployment dropdown
        const deploySelect = document.getElementById('savedModelSelect');
        deploySelect.innerHTML = '<option value="">-- Select saved model --</option>';
        
        // Update Backtesting dropdown
        const backtestSelect = document.getElementById('backtestModel');
        backtestSelect.innerHTML = '<option value="">Выберите обученную модель...</option>';
        
        result.data.forEach(model => {
            // Add to Deployment dropdown
            const deployOption = document.createElement('option');
            deployOption.value = model.filename;
            deployOption.textContent = `${model.model_type} - ${new Date(model.trained_at).toLocaleString()} (acc: ${model.accuracy.toFixed(3)})`;
            deployOption.dataset.metadata = JSON.stringify(model);
            deploySelect.appendChild(deployOption);
            
            // Add to Backtesting dropdown
            const backtestOption = document.createElement('option');
            backtestOption.value = model.filename;
            backtestOption.textContent = `${model.model_type} - ${new Date(model.trained_at).toLocaleString()} (acc: ${model.accuracy.toFixed(3)})`;
            backtestOption.dataset.metadata = JSON.stringify(model);
            backtestSelect.appendChild(backtestOption);
        });
    }
}

async function loadSelectedModel() {
    const select = document.getElementById('savedModelSelect');
    const filename = select.value;
    
    if (!filename) {
        alert('Please select a model to load');
        return;
    }
    
    const result = await apiRequest('/api/models/load', 'POST', {
        filename: filename
    });
    
    if (result.success) {
        modelTrained = true;
        document.getElementById('runBacktestBtn').disabled = false;
        document.getElementById('deployModelBtn').disabled = false;
        
        // Update metrics
        document.getElementById('metricAccuracy').textContent = result.data.accuracy.toFixed(3);
        document.getElementById('metricPrecision').textContent = result.data.precision.toFixed(3);
        document.getElementById('metricRecall').textContent = result.data.recall.toFixed(3);
        document.getElementById('metricF1').textContent = result.data.f1.toFixed(3);
        
        alert('Model loaded successfully');
    } else {
        alert(`Error loading model: ${result.message}`);
    }
}

// Backtesting
async function runBacktest() {
    const entryThreshold = parseFloat(document.getElementById('entryThreshold').value);
    const exitThreshold = parseFloat(document.getElementById('exitThreshold').value);
    const stopLoss = parseFloat(document.getElementById('stopLoss').value);
    const takeProfit = parseFloat(document.getElementById('takeProfit').value);
    const positionSize = parseFloat(document.getElementById('positionSize').value);
    
    const statusElement = document.getElementById('backtestStatus');
    statusElement.querySelector('.value').textContent = 'Running backtest...';
    
    const result = await apiRequest('/api/backtest/run', 'POST', {
        entry_threshold: entryThreshold,
        exit_threshold: exitThreshold,
        stop_loss: stopLoss,
        take_profit: takeProfit,
        position_size: positionSize
    });
    
    if (result.success) {
        backtestRun = true;
        statusElement.querySelector('.value').textContent = 'Backtest completed';
        
        // Update backtest metrics
        document.getElementById('backtestReturn').textContent = `${result.data.total_return > 0 ? '+' : ''}${result.data.total_return}%`;
        document.getElementById('backtestSharpe').textContent = result.data.sharpe_ratio.toFixed(2);
        document.getElementById('backtestDrawdown').textContent = `${result.data.max_drawdown}%`;
        document.getElementById('backtestWinRate').textContent = `${result.data.win_rate}%`;
        document.getElementById('backtestProfitFactor').textContent = result.data.profit_factor.toFixed(2);
        document.getElementById('backtestTrades').textContent = result.data.trade_count;
    } else {
        statusElement.querySelector('.value').textContent = `Error: ${result.message}`;
    }
}

// Deployment
async function deployModel() {
    const tradingMode = document.querySelector('input[name="tradingMode"]:checked').value;
    const positionSize = parseFloat(document.getElementById('deployPositionSize').value);
    const maxDailyLoss = parseFloat(document.getElementById('maxDailyLoss').value);
    
    const statusElement = document.getElementById('deploymentStatus');
    statusElement.querySelector('.value').textContent = 'Deploying...';
    
    const result = await apiRequest('/api/deployment/deploy', 'POST', {
        trading_mode: tradingMode,
        position_size: positionSize,
        max_daily_loss: maxDailyLoss
    });
    
    if (result.success) {
        modelDeployed = true;
        statusElement.querySelector('.value').textContent = `Model deployed in ${result.data.trading_mode} mode`;
        document.getElementById('deployModelBtn').disabled = true;
        document.getElementById('stopDeploymentBtn').disabled = false;
    } else {
        statusElement.querySelector('.value').textContent = `Error: ${result.message}`;
    }
}

async function stopDeployment() {
    const statusElement = document.getElementById('deploymentStatus');
    statusElement.querySelector('.value').textContent = 'Stopping...';
    
    const result = await apiRequest('/api/deployment/stop', 'POST');
    
    if (result.success) {
        modelDeployed = false;
        statusElement.querySelector('.value').textContent = 'Not deployed';
        document.getElementById('deployModelBtn').disabled = false;
        document.getElementById('stopDeploymentBtn').disabled = true;
    } else {
        statusElement.querySelector('.value').textContent = `Error: ${result.message}`;
    }
}
