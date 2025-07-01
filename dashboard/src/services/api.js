import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8001',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Agent API endpoints
export const agentAPI = {
  // Health check
  getHealth: () => api.get('/health'),
  
  // Agent management
  getAgents: () => api.get('/agents'),
  getAgent: (agentId) => api.get(`/agents/${agentId}`),
  createAgent: (agentData) => api.post('/agents', agentData),
  executeAgent: (agentId, taskData) => api.post(`/agents/${agentId}/execute`, taskData),
  deleteAgent: (agentId) => api.delete(`/agents/${agentId}`),
  
  // Agent types
  getAgentTypes: () => api.get('/agents/types'),
  
  // Batch operations
  createAgentsBatch: (agentsData) => api.post('/agents/batch', agentsData),
  
  // System status
  getSystemStatus: () => api.get('/system/status'),
};

// WebSocket service
export class WebSocketService {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.listeners = new Map();
    this.isConnected = false;
  }

  connect() {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8001/ws';
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.isConnected = false;
          this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      this.isConnected = false;
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
        this.connect().catch((error) => {
          console.error('Reconnection failed:', error);
        });
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  send(message) {
    if (this.ws && this.isConnected) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected');
    }
  }

  subscribe(messageType, callback) {
    if (!this.listeners.has(messageType)) {
      this.listeners.set(messageType, []);
    }
    this.listeners.get(messageType).push(callback);

    // Send subscription message
    this.send({
      type: 'subscribe',
      data: {
        message_types: [messageType]
      }
    });

    return () => {
      const callbacks = this.listeners.get(messageType);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    };
  }

  unsubscribe(messageType) {
    this.send({
      type: 'unsubscribe',
      data: {
        message_types: [messageType]
      }
    });
  }

  handleMessage(data) {
    const messageType = data.type;
    const callbacks = this.listeners.get(messageType);
    
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in WebSocket callback:', error);
        }
      });
    }
  }

  // Convenience methods for common operations
  requestAgentStatus(agentId = null) {
    this.send({
      type: 'agent_status',
      data: agentId ? { agent_id: agentId } : {}
    });
  }

  requestSystemStatus() {
    this.send({
      type: 'system_status',
      data: {}
    });
  }

  executeAgent(agentId, taskData) {
    this.send({
      type: 'agent_execution',
      data: {
        agent_id: agentId,
        task_data: taskData
      }
    });
  }
}

// Create singleton instance
export const wsService = new WebSocketService();

// Export default API instance
export default api; 