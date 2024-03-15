from pathlib import Path
import pandas as pd
from ema_workbench import Model, save_results


def save_households(households: pd.DataFrame, params: dict, random_seed: int):
    '''Save region households data to a CSV file.'''
    country = params['country']
    region = params['region']
    output_dir = Path(f'../experiments/{country}/households/')
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f'{region}_{random_seed}.csv'
    households.to_csv(file_path)


def save_experiment_results(country: str, return_period: int, model: Model, results: dict, n_scenarios: int, n_policies: int):
    """Saves experiment results to a file, taking into account if there was a conflict."""
    results_path = Path(f'../experiments/{country}')
    results_path.mkdir(parents=True, exist_ok=True)

    is_conflict = getattr(model.constants._data.get(
        'is_conflict'), 'value', False)

    conflict_str = ", conflict=True" if is_conflict else ""
    filename = f"return_period={return_period}, scenarios={n_scenarios}, policies={n_policies}{conflict_str}.tar.gz"
    save_results(results, results_path / filename)
