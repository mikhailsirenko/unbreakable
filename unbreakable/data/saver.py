from pathlib import Path
import pandas as pd


def save_households(households: pd.DataFrame, params: dict, random_seed: int):
    '''Save region households data to a CSV file.

    Args:
        households (pd.DataFrame): Households data.
        params (dict): A dictionary of parameters.
        random_seed (int): Random seed.

    Returns:
        None
    '''
    output_dir = Path(f'../results/{params["country"]}/households/')
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f'{params["region"]}_{random_seed}.csv'
    households.to_csv(file_path)
