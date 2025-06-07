// Initialize Socket.IO connection
const socket = io();

// DOM Elements
const cpuUsage = document.getElementById('cpuUsage');
const memoryUsage = document.getElementById('memoryUsage');
const diskUsage = document.getElementById('diskUsage');
const taskList = document.getElementById('taskList');
const alertList = document.getElementById('alertList');
const logContainer = document.getElementById('logContainer');
const refreshBtn = document.getElementById('refreshBtn');
const settingsBtn = document.getElementById('settingsBtn');
const newTaskBtn = document.getElementById('newTaskBtn');
const createTaskBtn = document.getElementById('createTaskBtn');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');

// Bootstrap Modals
const newTaskModal = new bootstrap.Modal(document.getElementById('newTaskModal'));
const settingsModal = new bootstrap.Modal(document.getElementById('settingsModal'));

// State
let tasks = [];
let alerts = [];
let logs = [];

// Event Listeners
refreshBtn.addEventListener('click', refreshDashboard);
settingsBtn.addEventListener('click', showSettings);
newTaskBtn.addEventListener('click', () => newTaskModal.show());
createTaskBtn.addEventListener('click', createTask);
saveSettingsBtn.addEventListener('click', saveSettings);

// Socket.IO Event Handlers
socket.on('connect', () => {
    console.log('Connected to server');
    socket.emit('start_monitoring');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
});

socket.on('metrics_update', (data) => {
    updateMetrics(data);
});

socket.on('task_update', (data) => {
    updateTask(data);
});

socket.on('alert', (data) => {
    addAlert(data);
});

socket.on('log', (data) => {
    addLog(data);
});

// Functions
async function refreshDashboard() {
    try {
        const [tasksRes, metricsRes, alertsRes, logsRes] = await Promise.all([
            fetch('/tasks'),
            fetch('/monitoring'),
            fetch('/errors'),
            fetch('/logs')
        ]);

        tasks = await tasksRes.json();
        const metrics = await metricsRes.json();
        alerts = await alertsRes.json();
        logs = await logsRes.json();

        updateTasksList();
        updateMetrics(metrics);
        updateAlertsList();
        updateLogs();
    } catch (error) {
        console.error('Error refreshing dashboard:', error);
        showError('Failed to refresh dashboard');
    }
}

function updateMetrics(data) {
    cpuUsage.textContent = `${data.cpu_usage}%`;
    memoryUsage.textContent = `${data.memory_usage}%`;
    diskUsage.textContent = `${data.disk_usage}%`;

    // Update colors based on thresholds
    updateMetricColor(cpuUsage, data.cpu_usage, 80);
    updateMetricColor(memoryUsage, data.memory_usage, 85);
    updateMetricColor(diskUsage, data.disk_usage, 90);
}

function updateMetricColor(element, value, threshold) {
    if (value >= threshold) {
        element.style.color = '#dc3545';
    } else if (value >= threshold * 0.8) {
        element.style.color = '#ffc107';
    } else {
        element.style.color = '#198754';
    }
}

function updateTasksList() {
    taskList.innerHTML = tasks.map(task => `
        <div class="card mb-2">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <h6 class="card-title">${task.type}</h6>
                        <p class="card-text">${task.description}</p>
                    </div>
                    <div>
                        <span class="badge bg-${getPriorityColor(task.priority)}">${task.priority}</span>
                        <span class="badge bg-${getStatusColor(task.status)}">${task.status}</span>
                    </div>
                </div>
                <div class="progress mt-2">
                    <div class="progress-bar" role="progressbar" style="width: ${task.progress}%"></div>
                </div>
            </div>
        </div>
    `).join('');
}

function updateAlertsList() {
    alertList.innerHTML = alerts.map(alert => `
        <div class="alert alert-${getAlertType(alert.severity)} alert-card">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <strong>${alert.title}</strong>
                    <p class="mb-0">${alert.message}</p>
                </div>
                <small>${new Date(alert.timestamp).toLocaleTimeString()}</small>
            </div>
        </div>
    `).join('');
}

function updateLogs() {
    logContainer.innerHTML = logs.map(log => `
        <div class="log-entry">
            <span class="text-muted">[${new Date(log.timestamp).toLocaleString()}]</span>
            <span class="text-${getLogLevelColor(log.level)}">${log.level}</span>
            <span>${log.message}</span>
        </div>
    `).join('');
    logContainer.scrollTop = logContainer.scrollHeight;
}

async function createTask() {
    const form = document.getElementById('newTaskForm');
    const formData = new FormData(form);
    const task = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/tasks', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(task)
        });

        if (response.ok) {
            newTaskModal.hide();
            form.reset();
            refreshDashboard();
        } else {
            showError('Failed to create task');
        }
    } catch (error) {
        console.error('Error creating task:', error);
        showError('Failed to create task');
    }
}

async function showSettings() {
    try {
        const response = await fetch('/config');
        const config = await response.json();
        
        const form = document.getElementById('settingsForm');
        form.innerHTML = generateSettingsForm(config);
        
        settingsModal.show();
    } catch (error) {
        console.error('Error loading settings:', error);
        showError('Failed to load settings');
    }
}

async function saveSettings() {
    const form = document.getElementById('settingsForm');
    const formData = new FormData(form);
    const config = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            settingsModal.hide();
            showSuccess('Settings saved successfully');
        } else {
            showError('Failed to save settings');
        }
    } catch (error) {
        console.error('Error saving settings:', error);
        showError('Failed to save settings');
    }
}

// Utility Functions
function getPriorityColor(priority) {
    const colors = {
        low: 'success',
        medium: 'warning',
        high: 'danger'
    };
    return colors[priority] || 'secondary';
}

function getStatusColor(status) {
    const colors = {
        pending: 'warning',
        running: 'primary',
        completed: 'success',
        failed: 'danger'
    };
    return colors[status] || 'secondary';
}

function getAlertType(severity) {
    const types = {
        critical: 'danger',
        warning: 'warning',
        info: 'info'
    };
    return types[severity] || 'info';
}

function getLogLevelColor(level) {
    const colors = {
        ERROR: 'danger',
        WARNING: 'warning',
        INFO: 'info',
        DEBUG: 'secondary'
    };
    return colors[level] || 'secondary';
}

function generateSettingsForm(config) {
    return Object.entries(config).map(([key, value]) => `
        <div class="mb-3">
            <label class="form-label">${key}</label>
            <input type="text" class="form-control" name="${key}" value="${value}">
        </div>
    `).join('');
}

function showError(message) {
    // Implement error notification
    console.error(message);
}

function showSuccess(message) {
    // Implement success notification
    console.log(message);
}

// Initial load
refreshDashboard(); 