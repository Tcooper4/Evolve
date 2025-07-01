import React, { useEffect, useState } from 'react';
import { Box, Typography, Button, CircularProgress } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import AgentTable from '../components/AgentTable';
import AgentForm from '../components/AgentForm';
import { agentAPI } from '../src/services/api';

function Agents() {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState(false);

  const fetchAgents = () => {
    setLoading(true);
    agentAPI.getAgents().then(res => {
      setAgents(res.data);
      setLoading(false);
    });
  };

  useEffect(() => {
    fetchAgents();
  }, []);

  const handleCreate = () => setOpen(true);
  const handleClose = () => setOpen(false);
  const handleCreated = () => {
    setOpen(false);
    fetchAgents();
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h4">Agents</Typography>
        <Button variant="contained" startIcon={<AddIcon />} onClick={handleCreate}>
          Create Agent
        </Button>
      </Box>
      {loading ? (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="40vh"><CircularProgress /></Box>
      ) : (
        <AgentTable agents={agents} />
      )}
      <AgentForm open={open} onClose={handleClose} onCreated={handleCreated} />
    </Box>
  );
}

export default Agents; 