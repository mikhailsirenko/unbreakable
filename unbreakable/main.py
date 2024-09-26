from unbreakable.experiments.config_handler import load_config, update_config
from unbreakable.experiments.model_setup import setup_model
from unbreakable.experiments.experiment_runner import run_experiments

from ema_workbench import ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)


def main():
    try:
        country = "Bangladesh"
        disaster_spec = [
            {"type": "flood", "event_time": 0, "return_period": None},
            # {"type": "flood", "event_time": 0, "return_period": 50},
            # {"type": "earthquake", "event_time": 0, "return_period": 100},
        ]

        if len(disaster_spec) == 1:
            return_period = disaster_spec[0]["return_period"]
        else:
            return_period = None

        config = load_config(country)
        config = update_config(config, disaster_spec)
        model = setup_model(config)

        experimental_setup = {
            "country": country,
            "disaster_spec": disaster_spec,
            "model": model,
            "return_period": return_period,
            "n_scenarios": 2,
            "n_policies": 0,
            "multiprocessing": False,
            "n_processes": 16,
        }

        run_experiments(experimental_setup)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
