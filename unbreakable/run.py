import yaml
import sys
from pathlib import Path
from unbreakable.model import *
from ema_workbench import *
ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == "__main__":
    try:
        country = sys.argv[1] if len(sys.argv) > 1 else 'Dominica'
        config = load_config(country)
        my_model = setup_model(config)

        n_scenarios = 100  # Number of scenarios
        n_policies = 0     # Number of policies

        run_experiments(my_model, n_scenarios, n_policies, country)

    except Exception as e:
        print(f"An error occurred: {e}")
