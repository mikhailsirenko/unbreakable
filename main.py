from experiments.config_manager import load_config
from experiments.model_setup import setup_model
from experiments.experiment_runner import run_experiments
from ema_workbench import ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)


def main():
    try:
        country = 'Dominica'
        disaster_type = 'hurricane'
        return_period = 100
        is_conflict = False
        config = load_config(country, return_period,
                             disaster_type, is_conflict)
        model = setup_model(config)

        experimental_setup = {
            'country': country,
            'return_period': return_period,
            'model': model,
            'n_scenarios': 2,
            'n_policies': 0,
            'multiprocessing': False,
            'n_processes': 12
        }

        run_experiments(experimental_setup)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
