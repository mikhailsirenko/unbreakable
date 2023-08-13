# The main file of the model. It runs the model with the parameters specified from the config.yaml.

import yaml
from pathlib import Path
from ema_workbench import (Model, Constant, CategoricalParameter, IntegerParameter, RealParameter,
                           ArrayOutcome, MultiprocessingEvaluator, ema_logging, perform_experiments, save_results)
from unbreakable.model import *
ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == "__main__":
    with open("../config/SaintLucia.yaml", "r") as f:
        config = yaml.safe_load(f)
    constants = config["constants"]
    uncertainties = config["uncertainties"]
    levers = config["levers"]

    my_model = Model(name="model", function=run_model)
    
    seed_start = 0
    seed_end = 1000000
    
    my_model.uncertainties = [IntegerParameter("random_seed", seed_start, seed_end)]\
        #   + [RealParameter(key, values[0], values[1]) for key, values in uncertainties.items()]

    my_model.constants = [Constant(key, values) for key, values in constants.items()]
    my_model.levers = [CategoricalParameter("my_policy", [value for key, value in levers.items()])]
    my_model.outcomes = [ArrayOutcome(district) for district in constants['districts']]

    n_scenarios = 36
    n_policies = 0
    
    # results = perform_experiments(models=my_model, scenarios=n_scenarios, policies=n_policies)

    with MultiprocessingEvaluator(my_model, n_processes=12) as evaluator:
        results = evaluator.perform_experiments(scenarios=n_scenarios, policies=n_policies)

    Path(f'../experiments/').mkdir(parents=True, exist_ok=True)

    save_results(results, f"../experiments/scenarios={n_scenarios}, policies={n_policies}.tar.gz")