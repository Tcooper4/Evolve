# API Module

This module provides the REST API endpoints for the agentic forecasting platform.

## Features

- RESTful API endpoints
- Request validation
- Response formatting
- Error handling
- Rate limiting
- Authentication middleware

## Structure

- `routes/`: API route definitions
- `middleware/`: Custom middleware (auth, logging, etc.)
- `schemas/`: Request/response schemas
- `controllers/`: Business logic for endpoints
- `utils/`: API-specific utilities

## Usage

```python
from api import create_app

app = create_app()
app.run(host="0.0.0.0", port=8000)
```

## API Documentation

API documentation is available at `/docs` when running the server.

## Testing

Run API tests:
```bash
pytest tests/api/
```

## Security

- JWT authentication
- Rate limiting
- Input validation
- CORS configuration
- Request sanitization 