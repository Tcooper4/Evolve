import pytest
from trading.web.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_get_portfolio(client):
    """Test the /portfolio endpoint."""
    response = client.get('/portfolio')
    assert response.status_code == 200
    assert 'portfolio_value' in response.json

def test_add_asset(client):
    """Test the /portfolio/add endpoint."""
    response = client.post('/portfolio/add', json={'asset': 'AAPL', 'quantity': 10, 'price': 150.0})
    assert response.status_code == 200
    assert response.json['message'] == 'Asset added successfully'

def test_get_var(client):
    """Test the /risk/var endpoint."""
    response = client.get('/risk/var')
    assert response.status_code == 400
    assert response.json['error'] == 'Risk manager not initialized'

def test_run_backtest(client):
    """Test the /backtest/run endpoint."""
    response = client.post('/backtest/run', json={'strategy': None, 'data': {}})
    assert response.status_code == 400
    assert response.json['error'] == 'Strategy not provided' 