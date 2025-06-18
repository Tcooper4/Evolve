// Global configuration for the dashboard

const config = {
  // API endpoints
  api: {
    baseUrl: process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000',
    wsUrl: process.env.REACT_APP_WS_URL || 'ws://localhost:5000',
    endpoints: {
      trades: '/api/trades',
      signals: '/api/signals',
      performance: '/api/performance',
      agents: '/api/agents',
      health: '/api/health'
    }
  },

  // Chart settings
  charts: {
    defaultHeight: 400,
    defaultWidth: '100%',
    colors: {
      primary: '#2196F3',
      secondary: '#FFC107',
      success: '#4CAF50',
      danger: '#F44336',
      warning: '#FF9800',
      info: '#00BCD4'
    },
    timeframes: ['1m', '5m', '15m', '1h', '4h', '1d', '1w'],
    defaultTimeframe: '1d'
  },

  // Table settings
  tables: {
    defaultPageSize: 10,
    pageSizeOptions: [10, 25, 50, 100],
    defaultSortField: 'timestamp',
    defaultSortOrder: 'desc'
  },

  // Agent settings
  agents: {
    refreshInterval: 5000, // 5 seconds
    statusColors: {
      active: '#4CAF50',
      idle: '#FFC107',
      error: '#F44336',
      offline: '#9E9E9E'
    }
  },

  // Feature flags
  features: {
    liveSignals: process.env.REACT_APP_ENABLE_LIVE_SIGNALS === 'true',
    agentControl: process.env.REACT_APP_ENABLE_AGENT_CONTROL === 'true',
    performanceTracking: process.env.REACT_APP_ENABLE_PERFORMANCE_TRACKING === 'true'
  },

  // UI settings
  ui: {
    theme: process.env.REACT_APP_THEME || 'light',
    language: process.env.REACT_APP_LANGUAGE || 'en',
    dateFormat: 'YYYY-MM-DD HH:mm:ss',
    numberFormat: {
      style: 'decimal',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }
  }
};

export default config; 