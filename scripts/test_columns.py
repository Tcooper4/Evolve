import sys

sys.path.insert(0, ".")
from trading.data.price_cache import get_history

df = get_history("AAPL", period="1y")
print("Columns:", list(df.columns))
print("Index name:", df.index.name)
print("Shape:", df.shape)
