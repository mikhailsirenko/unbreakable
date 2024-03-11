from unbreakable.model import load_config, setup_model, run_experiments
from ema_workbench import ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == "__main__":
    try:
        country = 'Nigeria'
        return_period = 100

        config = load_config(country, return_period)
        model = setup_model(config, replicator=False)

        experimental_setup = {
            'country': country,
            'return_period': return_period,
            'model': model,
            'n_scenarios': 12,
            'n_policies': 0,
            'multiprocessing': True,
            'n_processes': 12
        }

        run_experiments(experimental_setup)

    except Exception as e:
        print(f"An error occurred: {e}")
