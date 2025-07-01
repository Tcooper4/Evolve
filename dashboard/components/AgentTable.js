import React from 'react';
import { DataGrid } from '@mui/x-data-grid';
import { Box, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';

function AgentTable({ agents }) {
  const navigate = useNavigate();
  const columns = [
    { field: 'agent_id', headerName: 'Agent ID', width: 180 },
    { field: 'agent_type', headerName: 'Type', width: 140 },
    { field: 'status', headerName: 'Status', width: 120 },
    { field: 'capabilities', headerName: 'Capabilities', width: 200, valueGetter: (params) => params.row.capabilities?.join(', ') },
    { field: 'created_at', headerName: 'Created At', width: 180 },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 120,
      renderCell: (params) => (
        <Button variant="outlined" size="small" onClick={() => navigate(`/agents/${params.row.agent_id}`)}>
          Details
        </Button>
      ),
    },
  ];
  return (
    <Box sx={{ height: 500, width: '100%' }}>
      <DataGrid
        rows={agents.map((a) => ({ ...a, id: a.agent_id }))}
        columns={columns}
        pageSize={10}
        rowsPerPageOptions={[10, 25, 50]}
        disableSelectionOnClick
        autoHeight
      />
    </Box>
  );
}

export default AgentTable; 