"""This module runs the model with the parameters specified from the config yaml file.
We run the model using the Python package EMA Workbench.
In a nutshell, EMA Workbench provides a convenient way to run a model with many scenarios and policies.
It also allows running multiple models in parallel, which is very useful for computationally expensive models.
To get to know more about running a model with EMA Workbench, visit https://emaworkbench.readthedocs.io/en/latest/ema_documentation/index.html"""

import yaml
from pathlib import Path
from ema_workbench import (ReplicatorModel, Model, Constant, CategoricalParameter, IntegerParameter, RealParameter,
                           ArrayOutcome, MultiprocessingEvaluator, ema_logging, perform_experiments, save_results)
from unbreakable.model import *
ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == "__main__":
    # Load the config file with the parameters
    with open("../config/SaintLucia.yaml", "r") as f:
        config = yaml.safe_load(f)
    constants = config["constants"]
    uncertainties = config["uncertainties"]
    levers = config["levers"]

    # Initialize the EMA Workbench model
    my_model = Model(name="model", function=model)
    # my_model = ReplicatorModel(name="model", function=run_model)

    seed_start = 0
    seed_end = 1000000000

    # Unpack and initialise the uncertainties, constants, and levers
    my_model.uncertainties = [IntegerParameter("random_seed", seed_start, seed_end)]\
        #   + [RealParameter(key, values[0], values[1]) for key, values in uncertainties.items()]

    my_model.constants = [Constant(key, values)
                          for key, values in constants.items()]
    my_model.levers = [CategoricalParameter(
        "my_policy", [value for _, value in levers.items()])]

    # Specify the outcomes. Each outcome is an array for a single district.
    my_model.outcomes = [ArrayOutcome(district)
                         for district in constants['districts']]

    # Specify the number of scenarios and policies
    n_scenarios = 24
    n_policies = 0

    # results = perform_experiments(
    #     models=my_model, scenarios=n_scenarios, policies=n_policies)

    # Perform the experiments
    with MultiprocessingEvaluator(my_model, n_processes=12) as evaluator:
        results = evaluator.perform_experiments(
            scenarios=n_scenarios, policies=n_policies)

    # Save the results
    Path(f'../experiments/').mkdir(parents=True, exist_ok=True)

    save_results(
        results, f"../experiments/scenarios={n_scenarios}, policies={n_policies}.tar.gz")
