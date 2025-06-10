import React, { useState, useEffect } from 'react';
import { Typography, IconButton, Box } from '@mui/material';
import { Brightness4, Brightness7 } from '@mui/icons-material';
import Layout from '../components/Layout';
import Chart from '../components/Chart';
import DataTable from '../components/DataTable';
import StatusIndicator from '../components/StatusIndicator';
import { useTheme } from '../components/ThemeProvider';

const Dashboard = () => {
  const { mode, toggleTheme } = useTheme();
  const [systemMetrics, setSystemMetrics] = useState({
    cpu: {
      labels: [],
      datasets: [
        {
          label: 'CPU Usage',
          data: [],
          borderColor: '#2196F3',
          tension: 0.4,
        },
      ],
    },
    memory: {
      labels: [],
      datasets: [
        {
          label: 'Memory Usage',
          data: [],
          borderColor: '#4CAF50',
          tension: 0.4,
        },
      ],
    },
  });

  const [systemStatus, setSystemStatus] = useState({
    services: [
      { id: 1, name: 'API Service', status: 'running', uptime: '5d 3h' },
      { id: 2, name: 'Database', status: 'running', uptime: '5d 3h' },
      { id: 3, name: 'Cache', status: 'warning', uptime: '2h 15m' },
      { id: 4, name: 'Queue', status: 'running', uptime: '5d 3h' },
    ],
    alerts: [
      { id: 1, severity: 'warning', message: 'High memory usage on Cache service' },
      { id: 2, severity: 'info', message: 'System backup completed' },
    ],
  });

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date().toLocaleTimeString();
      
      setSystemMetrics((prev) => ({
        cpu: {
          ...prev.cpu,
          labels: [...prev.cpu.labels.slice(-19), now],
          datasets: [
            {
              ...prev.cpu.datasets[0],
              data: [...prev.cpu.datasets[0].data.slice(-19), Math.random() * 100],
            },
          ],
        },
        memory: {
          ...prev.memory,
          labels: [...prev.memory.labels.slice(-19), now],
          datasets: [
            {
              ...prev.memory.datasets[0],
              data: [...prev.memory.datasets[0].data.slice(-19), Math.random() * 100],
            },
          ],
        },
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const serviceColumns = [
    { field: 'name', headerName: 'Service Name' },
    { field: 'status', headerName: 'Status' },
    { field: 'uptime', headerName: 'Uptime' },
  ];

  const alertColumns = [
    { field: 'severity', headerName: 'Severity' },
    { field: 'message', headerName: 'Message' },
  ];

  return (
    <Layout theme={mode}>
      <Layout.Item xs={12}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4">System Dashboard</Typography>
          <IconButton onClick={toggleTheme} color="inherit">
            {mode === 'dark' ? <Brightness7 /> : <Brightness4 />}
          </IconButton>
        </Box>
      </Layout.Item>

      <Layout.Item xs={12} md={6}>
        <Chart
          title="CPU Usage"
          data={systemMetrics.cpu}
          height={300}
          theme={mode}
        />
      </Layout.Item>

      <Layout.Item xs={12} md={6}>
        <Chart
          title="Memory Usage"
          data={systemMetrics.memory}
          height={300}
          theme={mode}
        />
      </Layout.Item>

      <Layout.Item xs={12} md={6}>
        <Typography variant="h6" sx={{ mb: 2 }}>Service Status</Typography>
        <DataTable
          columns={serviceColumns}
          data={systemStatus.services}
          theme={mode}
        />
      </Layout.Item>

      <Layout.Item xs={12} md={6}>
        <Typography variant="h6" sx={{ mb: 2 }}>System Alerts</Typography>
        <DataTable
          columns={alertColumns}
          data={systemStatus.alerts}
          theme={mode}
        />
      </Layout.Item>

      <Layout.Item xs={12}>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          {systemStatus.services.map((service) => (
            <StatusIndicator
              key={service.id}
              status={service.status}
              label={service.name}
              theme={mode}
            />
          ))}
        </Box>
      </Layout.Item>
    </Layout>
  );
};

export default Dashboard; 