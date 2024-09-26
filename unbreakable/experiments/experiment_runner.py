from pathlib import Path
from typing import Dict, Any
from ema_workbench import perform_experiments, MultiprocessingEvaluator, save_results


def run_experiments(experimental_setup: Dict[str, Any]) -> None:
    """
    Run experiments with the specified setup using EMA Workbench and save the results.

    Args:
        experimental_setup (Dict[str, Any]): A dictionary containing the setup for the experiments.
    """
    country = experimental_setup["country"]
    disaster_spec = experimental_setup["disaster_spec"]
    model = experimental_setup["model"]
    return_period = experimental_setup["return_period"]
    n_scenarios = experimental_setup["n_scenarios"]
    n_policies = experimental_setup["n_policies"]
    multiprocessing = experimental_setup["multiprocessing"]
    n_processes = experimental_setup["n_processes"]

    if multiprocessing:
        with MultiprocessingEvaluator(model, n_processes=n_processes) as evaluator:
            results = evaluator.perform_experiments(
                scenarios=n_scenarios, policies=n_policies
            )
    else:
        results = perform_experiments(
            models=model, scenarios=n_scenarios, policies=n_policies
        )

    # If disaster spec has only one disaster, extract the disaster type
    if len(disaster_spec) == 1:
        disaster_type = disaster_spec[0]["type"]
        
    # Else, make a combined disaster type
    else:
        disaster_type = ""
        for i in range(len(disaster_spec)):
            disaster_type += disaster_spec[i]["type"]
            if i != len(disaster_spec) - 1:
                disaster_type += "_and_"

    results_path = Path(f"../results/{country}")
    results_path.mkdir(parents=True, exist_ok=True)
    filename = f"disaster_type={disaster_type}_return_period={return_period}_scenarios={n_scenarios}_policies={n_policies}.tar.gz"
    save_results(results, results_path / filename)
