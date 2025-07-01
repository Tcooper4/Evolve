import React, { useEffect, useState } from 'react';
import { Box, Typography, Card, CardContent, Grid, CircularProgress } from '@mui/material';
import { agentAPI } from '../src/services/api';
import { wsService } from '../src/services/api';

function Dashboard() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let unsubscribe = null;
    agentAPI.getSystemStatus().then(res => {
      setSystemStatus(res.data);
      setLoading(false);
    });
    wsService.connect().then(() => {
      unsubscribe = wsService.subscribe('system_status', (msg) => {
        setSystemStatus(msg.data);
      });
      wsService.requestSystemStatus();
    });
    return () => {
      if (unsubscribe) unsubscribe();
      wsService.disconnect();
    };
  }, []);

  if (loading || !systemStatus) {
    return <Box display="flex" justifyContent="center" alignItems="center" minHeight="40vh"><CircularProgress /></Box>;
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>System Overview</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6">System Health</Typography>
              <Typography color={systemStatus.system_status === 'healthy' ? 'green' : 'red'}>
                {systemStatus.system_status}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6">Total Agents</Typography>
              <Typography variant="h5">{systemStatus.total_agents}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6">Active Agents</Typography>
              <Typography variant="h5">{systemStatus.active_agents}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6">WebSocket Connections</Typography>
              <Typography variant="h5">{systemStatus.websocket_stats?.total_connections || 0}</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      {/* Placeholder for charts/metrics */}
      <Box mt={4}>
        <Typography variant="h6">Live Metrics & Charts (Coming Soon)</Typography>
      </Box>
    </Box>
  );
}

export default Dashboard; 