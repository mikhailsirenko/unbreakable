from unbreakable.model import *
from ema_workbench import *
ema_logging.log_to_stderr(ema_logging.INFO)


if __name__ == "__main__":
    try:
        country = 'Dominica'
        config = load_config(country)
        my_model = setup_model(config)

        n_scenarios = 50
        n_policies = 0
        multiprocessing = True

        run_experiments(my_model, n_scenarios, n_policies,
                        country, multiprocessing)

    except Exception as e:
        print(f"An error occurred: {e}")
