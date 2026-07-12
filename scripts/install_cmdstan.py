"""Install / repair CmdStan so Prophet uses a real Stan backend."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("install_cmdstan")


def _bypass_prophet_stub() -> None:
    try:
        import prophet as _prophet_pkg

        stan_root = Path(_prophet_pkg.__file__).resolve().parent / "stan_model"
        for stub in stan_root.glob("cmdstan-*"):
            if stub.is_dir() and not (stub / "makefile").exists() and not stub.name.endswith(".broken"):
                dest = stub.with_name(stub.name + ".broken")
                if not dest.exists():
                    stub.rename(dest)
                    log.info("Renamed incomplete Prophet CmdStan stub → %s", dest.name)
    except Exception as e:
        log.warning("stub bypass: %s", e)


def main() -> None:
    from cmdstanpy import cmdstan_path, install_cmdstan, set_cmdstan_path

    _bypass_prophet_stub()
    try:
        path = cmdstan_path()
        log.info("CmdStan already installed at %s", path)
    except Exception:
        log.info("CmdStan missing — installing (several minutes)…")
        try:
            install_cmdstan(overwrite=False)
        except TypeError:
            install_cmdstan()
        path = cmdstan_path()
    set_cmdstan_path(path)
    os.environ["CMDSTAN"] = path
    log.info("CmdStan ready at %s", path)

    from prophet import Prophet
    import pandas as pd

    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=60, freq="D"),
        "y": range(60),
    })
    m = Prophet()
    assert getattr(m, "stan_backend", None) is not None, "stan_backend still missing"
    m.fit(df)
    log.info("Prophet Stan backend OK (%s)", type(m.stan_backend).__name__)


if __name__ == "__main__":
    main()
