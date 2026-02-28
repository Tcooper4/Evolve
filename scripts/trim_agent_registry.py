# -*- coding: utf-8 -*-
"""
Trim agent_registry.json to remove agents moved to _dead_code during rationalization.
Run from repo root: python scripts/trim_agent_registry.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "data" / "agent_registry.json"

# Keys to remove (moved to _dead_code/agents/)
REMOVE_AGENTS = {
    "metalearneragent",
    "metalearningfeedbackagent",
    "metaresearchagent",
    "metatuneragent",
    "multimodalagent",
    "optimizeragent",
    "promptrouteragent",
    "rollingretrainingagent",
    "selfimprovingagent",
    "selftuningoptimizeragent",
    "taskdelegationagent",
    "walkforwardagent",
}


def main():
    if not REGISTRY_PATH.exists():
        print(f"Registry not found: {REGISTRY_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    agents = data.get("agents", {})
    removed = []
    for key in REMOVE_AGENTS:
        if key in agents:
            del agents[key]
            removed.append(key)
    data["agents"] = agents
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Removed {len(removed)} agents: {removed}")


if __name__ == "__main__":
    main()
