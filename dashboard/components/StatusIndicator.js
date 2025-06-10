import React from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';

const StatusIndicator = ({
  status,
  label,
  size = 'medium',
  theme = 'light',
  showLabel = true,
}) => {
  const getStatusColor = () => {
    switch (status.toLowerCase()) {
      case 'running':
      case 'active':
      case 'success':
        return '#4CAF50';
      case 'warning':
      case 'pending':
        return '#FFC107';
      case 'error':
      case 'failed':
      case 'stopped':
        return '#F44336';
      case 'loading':
        return '#2196F3';
      default:
        return '#9E9E9E';
    }
  };

  const getSize = () => {
    switch (size) {
      case 'small':
        return { indicator: 16, label: '0.75rem' };
      case 'large':
        return { indicator: 32, label: '1.25rem' };
      default:
        return { indicator: 24, label: '1rem' };
    }
  };

  const sizes = getSize();
  const color = getStatusColor();

  const containerStyles = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  };

  const labelStyles = {
    color: theme === 'dark' ? '#FFFFFF' : '#000000',
    fontSize: sizes.label,
  };

  const renderIndicator = () => {
    if (status.toLowerCase() === 'loading') {
      return (
        <CircularProgress
          size={sizes.indicator}
          thickness={4}
          sx={{ color }}
        />
      );
    }

    return (
      <Box
        sx={{
          width: sizes.indicator,
          height: sizes.indicator,
          borderRadius: '50%',
          backgroundColor: color,
          boxShadow: `0 0 8px ${color}`,
        }}
      />
    );
  };

  return (
    <Box sx={containerStyles}>
      {renderIndicator()}
      {showLabel && (
        <Typography sx={labelStyles} variant="body2">
          {label || status}
        </Typography>
      )}
    </Box>
  );
};

export default StatusIndicator; 