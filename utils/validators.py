import pandas as pd

class SignalSchemaValidator:
    """
    Validator for signal DataFrame schema.
    Required columns: index, 'Close', 'SignalType'. Optional: 'Confidence'.
    """
    REQUIRED_COLS = {"Close", "SignalType"}
    OPTIONAL_COLS = {"Confidence"}

    @staticmethod
    def validate(df: pd.DataFrame) -> bool:
        if not isinstance(df, pd.DataFrame):
            return False
        if not SignalSchemaValidator.REQUIRED_COLS.issubset(set(df.columns)):
            return False
        if df.index is None or getattr(df.index, 'isnull', lambda: False)().any():
            return False
        return True

    @staticmethod
    def assert_valid(df: pd.DataFrame):
        if not SignalSchemaValidator.validate(df):
            raise ValueError(f"Signal DataFrame does not meet schema requirements: must have columns {SignalSchemaValidator.REQUIRED_COLS} and non-null index.") 