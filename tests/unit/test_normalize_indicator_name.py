import pytest
from trading.utils.common import normalize_indicator_name

@pytest.mark.parametrize("original,expected", [
    ("macd 12-26-9", "MACD_12_26_9"),
    ("RSI", "RSI"),
    ("bollinger-band", "BOLLINGER_BAND"),
])
def test_normalize_indicator_name(original, expected):
    assert normalize_indicator_name(original) == expected
