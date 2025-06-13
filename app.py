from initialization import init_session_state
from multipage import MultiPage
from pages import forecast, strategy, backtest


def main() -> None:
    init_session_state()
    app = MultiPage()
    app.add_page("Forecast", forecast.app)
    app.add_page("Strategy", strategy.app)
    app.add_page("Backtest", backtest.app)
    app.run()


if __name__ == "__main__":
    main()
