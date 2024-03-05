from unbreakable.model import load_config, setup_model, run_experiments
from ema_workbench import ema_logging
ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == "__main__":
    try:
        # Load the configuration for the specified country
        country = 'Dominica'
        config = load_config(country)

        # Design of experiments
        n_scenarios = 1
        # n_replications = 10
        n_policies = 7
        multiprocessing = True
        n_processes = 7

        # my_model = setup_model(config, n_replications=n_replications)
        my_model = setup_model(config)

        run_experiments(my_model, n_scenarios, n_policies,
                        country, multiprocessing, n_processes)

    except Exception as e:
        print(f"An error occurred: {e}")
