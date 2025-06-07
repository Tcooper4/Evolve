class AlphaVantageProvider:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_data(self, symbol):
        """Fetch data from Alpha Vantage for the given symbol."""
        return {}

    def get_historical_data(self, symbol, interval):
        """Get historical data from Alpha Vantage for the given symbol and interval."""
        return {} 