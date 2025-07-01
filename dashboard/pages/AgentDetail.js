import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Box, Typography, CircularProgress } from '@mui/material';
import { agentAPI } from '../src/services/api';
import { wsService } from '../src/services/api';
import AgentStatusCard from '../components/AgentStatusCard';

function AgentDetail() {
  const { agentId } = useParams();
  const [agent, setAgent] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let unsubscribe = null;
    agentAPI.getAgent(agentId).then(res => {
      setAgent(res.data);
      setLoading(false);
    });
    wsService.connect().then(() => {
      unsubscribe = wsService.subscribe('agent_status', (msg) => {
        if (msg.data.agent_id === agentId) {
          setAgent(msg.data);
        }
      });
      wsService.requestAgentStatus(agentId);
    });
    return () => {
      if (unsubscribe) unsubscribe();
      wsService.disconnect();
    };
  }, [agentId]);

  if (loading || !agent) {
    return <Box display="flex" justifyContent="center" alignItems="center" minHeight="40vh"><CircularProgress /></Box>;
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>Agent Detail</Typography>
      <AgentStatusCard agent={agent} />
      {/* Placeholder for agent controls, logs, and metrics */}
      <Box mt={4}>
        <Typography variant="h6">Agent Controls & Metrics (Coming Soon)</Typography>
      </Box>
    </Box>
  );
}

export default AgentDetail; 