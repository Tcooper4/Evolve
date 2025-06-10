import React from 'react';
import { Box, useTheme, useMediaQuery } from '@mui/material';

const Layout = ({
  children,
  columns = 12,
  gap = '1rem',
  padding = '1rem',
  theme = 'light',
}) => {
  const muiTheme = useTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(muiTheme.breakpoints.down('md'));

  const getGridTemplateColumns = () => {
    if (isMobile) return '1fr';
    if (isTablet) return 'repeat(2, 1fr)';
    return `repeat(${columns}, 1fr)`;
  };

  const containerStyles = {
    display: 'grid',
    gridTemplateColumns: getGridTemplateColumns(),
    gap,
    padding,
    backgroundColor: theme === 'dark' ? '#121212' : '#F5F5F5',
    minHeight: '100vh',
    width: '100%',
  };

  return <Box sx={containerStyles}>{children}</Box>;
};

const GridItem = ({
  children,
  xs = 12,
  sm,
  md,
  lg,
  xl,
  theme = 'light',
  padding = '1rem',
  backgroundColor,
}) => {
  const muiTheme = useTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(muiTheme.breakpoints.down('md'));
  const isDesktop = useMediaQuery(muiTheme.breakpoints.down('lg'));

  const getGridColumn = () => {
    if (isMobile) return `span ${xs}`;
    if (isTablet) return `span ${sm || xs}`;
    if (isDesktop) return `span ${md || sm || xs}`;
    return `span ${lg || md || sm || xs}`;
  };

  const itemStyles = {
    gridColumn: getGridColumn(),
    padding,
    backgroundColor: backgroundColor || (theme === 'dark' ? '#1E1E1E' : '#FFFFFF'),
    borderRadius: '4px',
    boxShadow: theme === 'dark'
      ? '0 2px 4px rgba(0, 0, 0, 0.2)'
      : '0 2px 4px rgba(0, 0, 0, 0.1)',
  };

  return <Box sx={itemStyles}>{children}</Box>;
};

Layout.Item = GridItem;

export default Layout; 