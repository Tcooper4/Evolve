import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { Typography, Box, Paper, Grid } from '@mui/material';
import Layout from '../components/Layout';
import Chart from '../components/Chart';
import DataTable from '../components/DataTable';
import StatusIndicator from '../components/StatusIndicator';
import { useTheme } from '../components/ThemeProvider';

const ServiceDetails = () => {
  const { serviceId } = useParams();
  const { mode } = useTheme();
  const [serviceData, setServiceData] = useState({
    name: 'API Service',
    status: 'running',
    uptime: '5d 3h',
    metrics: {
      responseTime: {
        labels: [],
        datasets: [
          {
            label: 'Response Time (ms)',
            data: [],
            borderColor: '#2196F3',
            tension: 0.4,
          },
        ],
      },
      requests: {
        labels: [],
        datasets: [
          {
            label: 'Requests per Minute',
            data: [],
            borderColor: '#4CAF50',
            tension: 0.4,
          },
        ],
      },
    },
    logs: [
      { timestamp: '2024-02-20 10:00:00', level: 'info', message: 'Service started' },
      { timestamp: '2024-02-20 10:01:00', level: 'info', message: 'Connected to database' },
      { timestamp: '2024-02-20 10:02:00', level: 'warning', message: 'High memory usage detected' },
      { timestamp: '2024-02-20 10:03:00', level: 'error', message: 'Database connection timeout' },
    ],
    configuration: {
      port: 8080,
      maxConnections: 1000,
      timeout: 30,
      retryAttempts: 3,
      cacheEnabled: true,
    },
  });

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date().toLocaleTimeString();
      
      setServiceData((prev) => ({
        ...prev,
        metrics: {
          responseTime: {
            ...prev.metrics.responseTime,
            labels: [...prev.metrics.responseTime.labels.slice(-19), now],
            datasets: [
              {
                ...prev.metrics.responseTime.datasets[0],
                data: [...prev.metrics.responseTime.datasets[0].data.slice(-19), Math.random() * 1000],
              },
            ],
          },
          requests: {
            ...prev.metrics.requests,
            labels: [...prev.metrics.requests.labels.slice(-19), now],
            datasets: [
              {
                ...prev.metrics.requests.datasets[0],
                data: [...prev.metrics.requests.datasets[0].data.slice(-19), Math.random() * 100],
              },
            ],
          },
        },
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const logColumns = [
    { field: 'timestamp', headerName: 'Timestamp' },
    { field: 'level', headerName: 'Level' },
    { field: 'message', headerName: 'Message' },
  ];

  const configColumns = [
    { field: 'key', headerName: 'Setting' },
    { field: 'value', headerName: 'Value' },
  ];

  const configData = Object.entries(serviceData.configuration).map(([key, value]) => ({
    key,
    value: value.toString(),
  }));

  return (
    <Layout theme={mode}>
      <Layout.Item xs={12}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
          <Typography variant="h4">{serviceData.name}</Typography>
          <StatusIndicator
            status={serviceData.status}
            label={`Uptime: ${serviceData.uptime}`}
            size="large"
            theme={mode}
          />
        </Box>
      </Layout.Item>

      <Layout.Item xs={12} md={6}>
        <Chart
          title="Response Time"
          data={serviceData.metrics.responseTime}
          height={300}
          theme={mode}
          yAxisUnit="ms"
        />
      </Layout.Item>

      <Layout.Item xs={12} md={6}>
        <Chart
          title="Request Rate"
          data={serviceData.metrics.requests}
          height={300}
          theme={mode}
          yAxisUnit="/min"
        />
      </Layout.Item>

      <Layout.Item xs={12} md={6}>
        <Typography variant="h6" sx={{ mb: 2 }}>Recent Logs</Typography>
        <DataTable
          columns={logColumns}
          data={serviceData.logs}
          theme={mode}
        />
      </Layout.Item>

      <Layout.Item xs={12} md={6}>
        <Typography variant="h6" sx={{ mb: 2 }}>Configuration</Typography>
        <DataTable
          columns={configColumns}
          data={configData}
          theme={mode}
        />
      </Layout.Item>
    </Layout>
  );
};

export default ServiceDetails; 