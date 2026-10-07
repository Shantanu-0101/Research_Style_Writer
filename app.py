"""
Unified entry point — delegates to the selected scenario app.
Will be updated in Phase 5 when all scenarios are merged.
For now, defaults to Scenario 1.
"""
import sys
import os

# Default to Scenario 1
scenario = os.getenv("SCENARIO", "1")

if scenario == "1":
    from apps.app_scenario1 import main
elif scenario == "2":
    # from apps.app_scenario2 import main
    raise NotImplementedError("Scenario 2 not yet implemented")
elif scenario == "3":
    # from apps.app_scenario3 import main
    raise NotImplementedError("Scenario 3 not yet implemented")
else:
    raise ValueError(f"Unknown scenario: {scenario}")

if __name__ == "__main__":
    main()
