import React, { useState, useEffect } from 'react';
import { Dialog, DialogTitle, DialogContent, DialogActions, Button, TextField, MenuItem, Box } from '@mui/material';
import { agentAPI } from '../src/services/api';

function AgentForm({ open, onClose, onCreated }) {
  const [agentType, setAgentType] = useState('');
  const [config, setConfig] = useState('{}');
  const [agentTypes, setAgentTypes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    agentAPI.getAgentTypes().then(res => {
      setAgentTypes(res.data);
    });
  }, []);

  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    try {
      const configObj = config ? JSON.parse(config) : {};
      await agentAPI.createAgent({ agent_type: agentType, config: configObj });
      setLoading(false);
      onCreated();
    } catch (e) {
      setError(e.message || 'Failed to create agent');
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Create New Agent</DialogTitle>
      <DialogContent>
        <Box mb={2}>
          <TextField
            select
            label="Agent Type"
            value={agentType}
            onChange={e => setAgentType(e.target.value)}
            fullWidth
            margin="normal"
          >
            {agentTypes.map((type) => (
              <MenuItem key={type.agent_type} value={type.agent_type}>
                {type.agent_type}
              </MenuItem>
            ))}
          </TextField>
          <TextField
            label="Agent Config (JSON)"
            value={config}
            onChange={e => setConfig(e.target.value)}
            fullWidth
            margin="normal"
            multiline
            minRows={4}
            placeholder="{ }"
          />
          {error && <Box color="red" mt={1}>{error}</Box>}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>Cancel</Button>
        <Button onClick={handleSubmit} variant="contained" disabled={loading || !agentType}>
          {loading ? 'Creating...' : 'Create'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default AgentForm; 