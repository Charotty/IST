// Dashboard JavaScript
// Handles WebSocket connections and real-time data updates

// Global variables
let socket;
let portfolioChart;
let metricsChart;
let portfolioData = [];
let metricsData = [];

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeSocket();
    initializeCharts();
    loadInitialData();
});

// Initialize WebSocket connection
function initializeSocket() {
    socket = io();
    
    socket.on('connect', function() {
        console.log('Connected to dashboard server');
        updateSystemStatus('connected', 'Connected');
        socket.emit('subscribe', { channels: ['prices', 'orders', 'portfolio', 'metrics'] });
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from dashboard server');
        updateSystemStatus('disconnected', 'Disconnected');
    });
    
    socket.on('price_update', function(data) {
        updatePrices(data);
    });
    
    socket.on('orders_update', function(data) {
        updateOrders(data);
    });
    
    socket.on('portfolio_update', function(data) {
        updatePortfolio(data);
    });
    
    socket.on('metrics_update', function(data) {
        updateMetrics(data);
    });
    
    socket.on('system_health', function(data) {
        updateSystemHealth(data);
    });
}

// Initialize charts
function initializeCharts() {
    // Portfolio Chart
    const portfolioCtx = document.getElementById('portfolioChart').getContext('2d');
    portfolioChart = new Chart(portfolioCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: '#00d9ff',
                backgroundColor: 'rgba(0, 217, 255, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: 'rgba(255, 255, 255, 0.8)' }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                }
            }
        }
    });
    
    // Metrics Chart
    const metricsCtx = document.getElementById('metricsChart').getContext('2d');
    metricsChart = new Chart(metricsCtx, {
        type: 'bar',
        data: {
            labels: ['Win Rate', 'Sharpe', 'Drawdown'],
            datasets: [{
                label: 'Metrics',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(0, 255, 136, 0.6)',
                    'rgba(0, 217, 255, 0.6)',
                    'rgba(255, 68, 68, 0.6)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: 'rgba(255, 255, 255, 0.8)' }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                }
            }
        }
    });
}

// Load initial data from API
async function loadInitialData() {
    try {
        // Load prices
        const pricesResponse = await fetch('/api/prices');
        const pricesData = await pricesResponse.json();
        updatePrices(pricesData);
        
        // Load orders
        const ordersResponse = await fetch('/api/orders');
        const ordersData = await ordersResponse.json();
        updateOrders(ordersData);
        
        // Load portfolio
        const portfolioResponse = await fetch('/api/portfolio');
        const portfolioData = await portfolioResponse.json();
        updatePortfolio(portfolioData);
        
        // Load metrics
        const metricsResponse = await fetch('/api/metrics');
        const metricsData = await metricsResponse.json();
        updateMetrics(metricsData);
        
        // Load system health
        const healthResponse = await fetch('/api/system-health');
        const healthData = await healthResponse.json();
        updateSystemHealth(healthData);
        
    } catch (error) {
        console.error('Error loading initial data:', error);
    }
}

// Update price displays
function updatePrices(prices) {
    for (const [symbol, data] of Object.entries(prices)) {
        const symbolId = symbol.replace('/', '').toLowerCase();
        const priceElement = document.getElementById(`${symbolId}-price`);
        const changeElement = document.getElementById(`${symbolId}-change`);
        
        if (priceElement) {
            priceElement.textContent = `$${data.price.toFixed(2)}`;
            
            // Simulate price change (in real implementation, calculate from previous price)
            const change = (Math.random() - 0.5) * 2;
            changeElement.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
            changeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
        }
    }
}

// Update orders list
function updateOrders(orders) {
    const activityList = document.getElementById('activityList');
    if (!activityList) return;
    
    activityList.innerHTML = '';
    
    orders.slice(0, 10).forEach(order => {
        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        
        const timestamp = new Date(order.dashboard_timestamp || Date.now());
        const timeString = timestamp.toLocaleTimeString();
        
        activityItem.innerHTML = `
            <span class="activity-message">${order.symbol || 'Unknown'} - ${order.side || 'Unknown'} ${order.amount || 0}</span>
            <span class="activity-time">${timeString}</span>
        `;
        
        activityList.appendChild(activityItem);
    });
}

// Update portfolio display
function updatePortfolio(portfolio) {
    // Update portfolio chart
    const now = new Date();
    const timeLabel = now.toLocaleTimeString();
    
    portfolioChart.data.labels.push(timeLabel);
    portfolioChart.data.datasets[0].data.push(portfolio.total_value || 0);
    
    // Keep only last 20 data points
    if (portfolioChart.data.labels.length > 20) {
        portfolioChart.data.labels.shift();
        portfolioChart.data.datasets[0].data.shift();
    }
    
    portfolioChart.update('none');
}

// Update metrics display
function updateMetrics(metrics) {
    // Update stat cards
    document.getElementById('totalTrades').textContent = metrics.total_trades || 0;
    document.getElementById('winRate').textContent = `${(metrics.win_rate * 100).toFixed(1)}%`;
    document.getElementById('sharpeRatio').textContent = metrics.sharpe_ratio.toFixed(2);
    document.getElementById('maxDrawdown').textContent = `${(metrics.max_drawdown * 100).toFixed(1)}%`;
    
    // Update metrics chart
    metricsChart.data.datasets[0].data = [
        metrics.win_rate * 100 || 0,
        metrics.sharpe_ratio || 0,
        metrics.max_drawdown * 100 || 0
    ];
    metricsChart.update('none');
}

// Update system health
function updateSystemHealth(health) {
    const indicator = document.querySelector('.status-indicator');
    const text = document.querySelector('.status-text');
    
    if (health.status === 'running' || health.status === 'healthy') {
        indicator.classList.add('healthy');
        text.textContent = 'Running';
    } else {
        indicator.classList.remove('healthy');
        text.textContent = health.status || 'Unknown';
    }
}

// Update system status indicator
function updateSystemStatus(status, message) {
    const indicator = document.querySelector('.status-indicator');
    const text = document.querySelector('.status-text');
    
    if (status === 'connected') {
        indicator.classList.add('healthy');
        text.textContent = message;
    } else {
        indicator.classList.remove('healthy');
        text.textContent = message;
    }
}

// Format number with currency
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

// Format percentage
function formatPercentage(value) {
    return `${(value * 100).toFixed(2)}%`;
}
