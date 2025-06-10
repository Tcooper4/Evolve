import React, { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  Paper,
  Switch,
  FormControlLabel,
  TextField,
  Button,
  Divider,
  Alert,
  Snackbar,
} from '@mui/material';
import Layout from '../components/Layout';
import { useTheme } from '../components/ThemeProvider';
import { useAuth } from '../components/AuthProvider';

const Settings = () => {
  const { mode, toggleTheme } = useTheme();
  const { user, hasPermission } = useAuth();
  const [settings, setSettings] = useState({
    notifications: {
      email: true,
      slack: false,
      sms: false,
    },
    refreshInterval: 30,
    dataRetention: 30,
    autoLogout: 60,
  });
  const [systemSettings, setSystemSettings] = useState({
    maxConnections: 1000,
    timeout: 30,
    retryAttempts: 3,
    cacheEnabled: true,
  });
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'success',
  });

  useEffect(() => {
    // Load settings from API
    const loadSettings = async () => {
      try {
        const response = await fetch('/api/settings');
        if (response.ok) {
          const data = await response.json();
          setSettings(data.userSettings);
          setSystemSettings(data.systemSettings);
        }
      } catch (error) {
        console.error('Failed to load settings:', error);
      }
    };

    loadSettings();
  }, []);

  const handleNotificationChange = (type) => (event) => {
    setSettings((prev) => ({
      ...prev,
      notifications: {
        ...prev.notifications,
        [type]: event.target.checked,
      },
    }));
  };

  const handleSettingChange = (field) => (event) => {
    setSettings((prev) => ({
      ...prev,
      [field]: event.target.value,
    }));
  };

  const handleSystemSettingChange = (field) => (event) => {
    setSystemSettings((prev) => ({
      ...prev,
      [field]: event.target.value,
    }));
  };

  const saveSettings = async () => {
    try {
      const response = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userSettings: settings,
          systemSettings,
        }),
      });

      if (response.ok) {
        setNotification({
          open: true,
          message: 'Settings saved successfully',
          severity: 'success',
        });
      } else {
        throw new Error('Failed to save settings');
      }
    } catch (error) {
      setNotification({
        open: true,
        message: 'Failed to save settings',
        severity: 'error',
      });
    }
  };

  return (
    <Layout theme={mode}>
      <Layout.Item xs={12}>
        <Typography variant="h4" sx={{ mb: 3 }}>
          Settings
        </Typography>
      </Layout.Item>

      <Layout.Item xs={12} md={6}>
        <Paper
          sx={{
            p: 3,
            backgroundColor: mode === 'dark' ? '#1E1E1E' : '#FFFFFF',
          }}
        >
          <Typography variant="h6" sx={{ mb: 2 }}>
            Notifications
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={settings.notifications.email}
                onChange={handleNotificationChange('email')}
              />
            }
            label="Email Notifications"
          />
          <FormControlLabel
            control={
              <Switch
                checked={settings.notifications.slack}
                onChange={handleNotificationChange('slack')}
              />
            }
            label="Slack Notifications"
          />
          <FormControlLabel
            control={
              <Switch
                checked={settings.notifications.sms}
                onChange={handleNotificationChange('sms')}
              />
            }
            label="SMS Notifications"
          />
        </Paper>
      </Layout.Item>

      <Layout.Item xs={12} md={6}>
        <Paper
          sx={{
            p: 3,
            backgroundColor: mode === 'dark' ? '#1E1E1E' : '#FFFFFF',
          }}
        >
          <Typography variant="h6" sx={{ mb: 2 }}>
            Preferences
          </Typography>
          <TextField
            fullWidth
            label="Refresh Interval (seconds)"
            type="number"
            value={settings.refreshInterval}
            onChange={handleSettingChange('refreshInterval')}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Data Retention (days)"
            type="number"
            value={settings.dataRetention}
            onChange={handleSettingChange('dataRetention')}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Auto Logout (minutes)"
            type="number"
            value={settings.autoLogout}
            onChange={handleSettingChange('autoLogout')}
            margin="normal"
          />
        </Paper>
      </Layout.Item>

      {hasPermission('admin') && (
        <>
          <Layout.Item xs={12}>
            <Divider sx={{ my: 3 }} />
            <Typography variant="h5" sx={{ mb: 3 }}>
              System Settings
            </Typography>
          </Layout.Item>

          <Layout.Item xs={12} md={6}>
            <Paper
              sx={{
                p: 3,
                backgroundColor: mode === 'dark' ? '#1E1E1E' : '#FFFFFF',
              }}
            >
              <Typography variant="h6" sx={{ mb: 2 }}>
                Performance
              </Typography>
              <TextField
                fullWidth
                label="Max Connections"
                type="number"
                value={systemSettings.maxConnections}
                onChange={handleSystemSettingChange('maxConnections')}
                margin="normal"
              />
              <TextField
                fullWidth
                label="Timeout (seconds)"
                type="number"
                value={systemSettings.timeout}
                onChange={handleSystemSettingChange('timeout')}
                margin="normal"
              />
              <TextField
                fullWidth
                label="Retry Attempts"
                type="number"
                value={systemSettings.retryAttempts}
                onChange={handleSystemSettingChange('retryAttempts')}
                margin="normal"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={systemSettings.cacheEnabled}
                    onChange={(e) =>
                      handleSystemSettingChange('cacheEnabled')({
                        target: { value: e.target.checked },
                      })
                    }
                  />
                }
                label="Enable Cache"
              />
            </Paper>
          </Layout.Item>
        </>
      )}

      <Layout.Item xs={12}>
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="contained"
            onClick={saveSettings}
            sx={{
              backgroundColor: mode === 'dark' ? '#2196F3' : '#1976D2',
              '&:hover': {
                backgroundColor: mode === 'dark' ? '#1976D2' : '#1565C0',
              },
            }}
          >
            Save Settings
          </Button>
        </Box>
      </Layout.Item>

      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={() => setNotification((prev) => ({ ...prev, open: false }))}
      >
        <Alert
          onClose={() => setNotification((prev) => ({ ...prev, open: false }))}
          severity={notification.severity}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Layout>
  );
};

export default Settings; 