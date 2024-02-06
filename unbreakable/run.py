from unbreakable.model import *
from ema_workbench import *
ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == "__main__":
    try:
        country = 'Nigeria'
        config = load_config(country)
        my_model = setup_model(config)

        n_scenarios = 3
        n_policies = 0
        multiprocessing = False
        n_processes = 12

        run_experiments(my_model, n_scenarios, n_policies,
                        country, multiprocessing, n_processes)

    except Exception as e:
        print(f"An error occurred: {e}")
