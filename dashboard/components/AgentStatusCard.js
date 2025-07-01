import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Grid } from '@mui/material';

function AgentStatusCard({ agent }) {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6">Agent ID: {agent.agent_id}</Typography>
        <Box mb={1}>
          <Chip label={agent.status} color={agent.status === 'active' ? 'success' : 'default'} />
        </Box>
        <Typography variant="body2">Capabilities: {agent.capabilities?.join(', ')}</Typography>
        <Typography variant="body2">Last Execution: {agent.last_execution || 'N/A'}</Typography>
        {agent.performance_metrics && (
          <Box mt={2}>
            <Typography variant="subtitle1">Performance Metrics:</Typography>
            <Grid container spacing={1}>
              {Object.entries(agent.performance_metrics).map(([key, value]) => (
                <Grid item key={key} xs={6} sm={4} md={3}>
                  <Box bgcolor="#f5f5f5" p={1} borderRadius={1} textAlign="center">
                    <Typography variant="caption">{key}</Typography>
                    <Typography variant="body2">{value}</Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default AgentStatusCard; 