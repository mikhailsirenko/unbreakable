from pathlib import Path
from ema_workbench import perform_experiments, MultiprocessingEvaluator, save_results, Model


def run_experiments(experimental_setup: dict) -> None:
    '''Run experiments with the specified setup with the use of EMA Workbench and save the results.
    
    Args:
        experimental_setup (dict): A dictionary containing the setup for the experiments.

    Returns:
        None
    '''
    country = experimental_setup['country']
    return_period = experimental_setup['return_period']
    model = experimental_setup['model']
    n_scenarios = experimental_setup['n_scenarios']
    n_policies = experimental_setup['n_policies']
    multiprocessing = experimental_setup['multiprocessing']
    n_processes = experimental_setup['n_processes']

    if multiprocessing:
        with MultiprocessingEvaluator(model, n_processes=n_processes) as evaluator:
            results = evaluator.perform_experiments(
                scenarios=n_scenarios, policies=n_policies)
    else:
        results = perform_experiments(
            models=model, scenarios=n_scenarios, policies=n_policies)

    save_experiment_results(country, return_period, model,
                            results, n_scenarios, n_policies)


def save_experiment_results(country: str, return_period: int, model: Model, results: dict, n_scenarios: int, n_policies: int):
    """Saves experiment results to a file, taking into account if there was a conflict."""
    results_path = Path(f'../results/{country}')
    results_path.mkdir(parents=True, exist_ok=True)

    is_conflict = getattr(model.constants._data.get(
        'is_conflict'), 'value', False)

    conflict_str = ", conflict=True" if is_conflict else ""
    filename = f"return_period={return_period}, scenarios={n_scenarios}, policies={n_policies}{conflict_str}.tar.gz"
    save_results(results, results_path / filename)
