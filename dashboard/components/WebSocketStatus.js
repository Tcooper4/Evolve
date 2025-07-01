import React, { useEffect, useState } from 'react';
import { Box, Tooltip } from '@mui/material';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';
import { wsService } from '../src/services/api';

function WebSocketStatus() {
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    wsService.connect().then(() => setConnected(true));
    wsService.ws.onclose = () => setConnected(false);
    return () => wsService.disconnect();
  }, []);

  return (
    <Tooltip title={connected ? 'WebSocket Connected' : 'WebSocket Disconnected'}>
      <Box ml={2} display="flex" alignItems="center">
        <FiberManualRecordIcon sx={{ color: connected ? 'green' : 'red', fontSize: 18 }} />
      </Box>
    </Tooltip>
  );
}

export default WebSocketStatus; 