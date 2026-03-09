import os
import re
import sys
import traceback


def main() -> None:
    sys.path.insert(0, os.getcwd())

    try:
        from agents.llm.agent import get_prompt_agent

        agent = get_prompt_agent()
        if not agent:
            print("[SKIP] No API key configured or prompt agent unavailable")
            sys.exit(0)

        # Simulate a forecast query
        response = agent.process_prompt("Give me a 7-day forecast for AAPL")
        msg = getattr(response, "message", None) or str(response)
        print(f"Response preview: {msg[:300]}")

        # Look for explicit dollar prices in the response
        prices = re.findall(r"\$\d{2,5}\.\d{2}", msg)
        if prices:
            print(f"[OK] Found price references: {prices}")
        else:
            print(
                "[WARN] No specific dollar prices found in response — chat may not be routing through forecast models"
            )
    except Exception as e:
        print(f"[FAIL] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

