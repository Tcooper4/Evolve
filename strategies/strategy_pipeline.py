import numpy as np
import pandas as pd
from typing import List, Callable, Dict, Any, Optional

# --- Individual Strategy Functions ---
def rsi_strategy(data: pd.DataFrame, window: int = 14, overbought: float = 70, oversold: float = 30) -> pd.Series:
    """RSI strategy: 1 for buy, -1 for sell, 0 for hold."""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    signals = pd.Series(0, index=data.index)
    signals[rsi > overbought] = -1
    signals[rsi < oversold] = 1
    return signals

def macd_strategy(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD strategy: 1 for buy, -1 for sell, 0 for hold."""
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    signals = pd.Series(0, index=data.index)
    signals[macd > macd_signal] = 1
    signals[macd < macd_signal] = -1
    return signals

def bollinger_strategy(data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger Bands strategy: 1 for buy, -1 for sell, 0 for hold."""
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    signals = pd.Series(0, index=data.index)
    signals[data['close'] > upper] = -1
    signals[data['close'] < lower] = 1
    return signals

def sma_strategy(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """SMA crossover strategy: 1 for buy, -1 for sell, 0 for hold."""
    sma = data['close'].rolling(window=window).mean()
    signals = pd.Series(0, index=data.index)
    signals[data['close'] > sma] = 1
    signals[data['close'] < sma] = -1
    return signals

# --- Signal Combination Logic ---
def combine_signals(
    signals_list: List[pd.Series],
    mode: str = 'intersection',
    weights: Optional[List[float]] = None
) -> pd.Series:
    """
    Combine multiple signal series.
    Modes:
        - 'union': signal is 1 if any input is 1, -1 if any is -1, else 0
        - 'intersection': signal is 1 if all are 1, -1 if all are -1, else 0
        - 'weighted': weighted sum of signals, sign determines output
    """
    if not signals_list:
        raise ValueError("signals_list must not be empty")
    signals_matrix = np.vstack([s.fillna(0).values for s in signals_list])
    if mode == 'union':
        buy = (signals_matrix == 1).any(axis=0)
        sell = (signals_matrix == -1).any(axis=0)
        result = np.zeros(signals_matrix.shape[1])
        result[buy] = 1
        result[sell] = -1
        return pd.Series(result, index=signals_list[0].index)
    elif mode == 'intersection':
        buy = (signals_matrix == 1).all(axis=0)
        sell = (signals_matrix == -1).all(axis=0)
        result = np.zeros(signals_matrix.shape[1])
        result[buy] = 1
        result[sell] = -1
        return pd.Series(result, index=signals_list[0].index)
    elif mode == 'weighted':
        if weights is None:
            weights = [1.0] * len(signals_list)
        weights = np.array(weights)
        weighted_sum = np.dot(weights, signals_matrix)
        result = np.sign(weighted_sum)
        return pd.Series(result, index=signals_list[0].index)
    else:
        raise ValueError(f"Unknown mode: {mode}")

# --- Example Usage ---
# data = pd.DataFrame({'close': ...})
# rsi_signals = rsi_strategy(data)
# macd_signals = macd_strategy(data)
# combo = combine_signals([rsi_signals, macd_signals], mode='intersection')

# --- For UI Integration ---
STRATEGY_FUNCTIONS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    'RSI': rsi_strategy,
    'MACD': macd_strategy,
    'Bollinger': bollinger_strategy,
    'SMA': sma_strategy,
}

COMBINE_MODES = ['union', 'intersection', 'weighted']

def get_strategy_names() -> List[str]:
    return list(STRATEGY_FUNCTIONS.keys())

def get_combine_modes() -> List[str]:
    return COMBINE_MODES 