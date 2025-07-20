import asyncio
import logging
import signal
import sys
import traceback

logger = logging.getLogger("runtime_initializer")

shutdown_event = asyncio.Event()

def handle_signal(signame):
    logger.info(f"Received signal {signame}, initiating graceful shutdown...")
    shutdown_event.set()

def setup_signal_handlers():
    for signame in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signame):
            signum = getattr(signal, signame)
            signal.signal(signum, lambda s, f: handle_signal(signame))

async def main():
    logger.info("Runtime initializer started.")
    setup_signal_handlers()
    try:
        # Main async logic here
        await shutdown_event.wait()
        logger.info("Shutdown event received. Cleaning up...")
    except Exception as e:
        logger.exception("Exception in runtime initializer:")
        traceback.print_exc()
        sys.exit(1)
    logger.info("Graceful shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception("Fatal error in runtime initializer:")
        traceback.print_exc()
        sys.exit(1)
