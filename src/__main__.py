# The main file of the model. It runs the model with the parameters specified from the config.yaml.

import yaml
from ema_workbench import (Model, Constant, CategoricalParameter, IntegerParameter, RealParameter,
                           ArrayOutcome, MultiprocessingEvaluator, ema_logging, perform_experiments, save_results)
from src.model import *
ema_logging.log_to_stderr(ema_logging.INFO)

my_model = Model(name="model", function=run_model)

if __name__ == "__main__":
    # Load config from yaml file
    with open("../config/main.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Each of the parameters is a dict
    constants = config["constants"]
    uncertainties = config["uncertainties"]
    policies = config["policies"]

    # Specify the EMA Workbench model
    my_model.constants = [Constant(key, values) for key, values in constants.items()]

    # Specify seed ranges. For EDA more seeds, for sensitivity less seeds, since we need to aggregate across seeds
    seed_start = 0
    seed_end = 1000000
    my_model.uncertainties = [IntegerParameter("random_seed", seed_start, seed_end)]\
        #   + [RealParameter(key, values[0], values[1]) for key, values in uncertainties.items()]

    my_model.levers = [CategoricalParameter("my_policy", [value for key, value in policies.items()])]

    # We store the outcomes of interest by district. To see which exact outcomes are stored, check the `get_outcomes` function in `write.py`
    my_model.outcomes = [ArrayOutcome(district) for district in constants['districts']]

    # Specify the number of scenarios & policies
    n_scenarios = 1
    n_policies = 0

    # Run the model on a single core
    results = perform_experiments(models=my_model, scenarios=n_scenarios, policies=n_policies)

    # Run the model on multiple cores
    # with MultiprocessingEvaluator(my_model, n_processes=12) as evaluator:
    #     results = evaluator.perform_experiments(scenarios=n_scenarios, policies=n_policies)

    # Save results as tar.gz file
    add_income_loss = constants["add_income_loss"]
    f = f"../results/scenarios={n_scenarios}, policies={n_policies}, income_loss={add_income_loss}.tar.gz"
    save_results(results, f)
