import React, { useState } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
} from '@mui/material';
import { useAuth } from '../components/AuthProvider';
import { useTheme } from '../components/ThemeProvider';

const Login = () => {
  const { login } = useAuth();
  const { mode } = useTheme();
  const [credentials, setCredentials] = useState({
    username: '',
    password: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setCredentials((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const success = await login(credentials);
      if (!success) {
        setError('Invalid username or password');
      }
    } catch (error) {
      setError('An error occurred during login');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: mode === 'dark' ? '#121212' : '#F5F5F5',
      }}
    >
      <Paper
        elevation={3}
        sx={{
          p: 4,
          width: '100%',
          maxWidth: 400,
          backgroundColor: mode === 'dark' ? '#1E1E1E' : '#FFFFFF',
        }}
      >
        <Typography
          variant="h4"
          component="h1"
          gutterBottom
          sx={{
            textAlign: 'center',
            color: mode === 'dark' ? '#FFFFFF' : '#000000',
          }}
        >
          Login
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Username"
            name="username"
            value={credentials.username}
            onChange={handleChange}
            margin="normal"
            required
            autoFocus
            sx={{
              '& .MuiInputLabel-root': {
                color: mode === 'dark' ? '#FFFFFF' : '#000000',
              },
              '& .MuiOutlinedInput-root': {
                color: mode === 'dark' ? '#FFFFFF' : '#000000',
                '& fieldset': {
                  borderColor: mode === 'dark' ? '#FFFFFF' : '#000000',
                },
              },
            }}
          />

          <TextField
            fullWidth
            label="Password"
            name="password"
            type="password"
            value={credentials.password}
            onChange={handleChange}
            margin="normal"
            required
            sx={{
              '& .MuiInputLabel-root': {
                color: mode === 'dark' ? '#FFFFFF' : '#000000',
              },
              '& .MuiOutlinedInput-root': {
                color: mode === 'dark' ? '#FFFFFF' : '#000000',
                '& fieldset': {
                  borderColor: mode === 'dark' ? '#FFFFFF' : '#000000',
                },
              },
            }}
          />

          <Button
            type="submit"
            fullWidth
            variant="contained"
            disabled={loading}
            sx={{
              mt: 3,
              mb: 2,
              backgroundColor: mode === 'dark' ? '#2196F3' : '#1976D2',
              '&:hover': {
                backgroundColor: mode === 'dark' ? '#1976D2' : '#1565C0',
              },
            }}
          >
            {loading ? <CircularProgress size={24} /> : 'Login'}
          </Button>
        </form>
      </Paper>
    </Box>
  );
};

export default Login; 