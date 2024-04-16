import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def read_data_file(file_path: Path, file_type: str = 'csv', **kwargs) -> pd.DataFrame:
    '''
    Generic function to read a data file.

    Args:
        file_path (Path): Path object to the file.
        file_type (str): Type of the file ('xlsx' or 'csv').
        **kwargs: Additional keyword arguments for pandas read functions.

    Returns:
        pd.DataFrame: Data from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    '''
    try:
        if file_type == 'xlsx':
            return pd.read_excel(file_path, **kwargs)
        elif file_type == 'csv':
            return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        logging.error(f"Error reading data from {file_path}: {e}")
        raise


def read_data(country: str, is_conflict: bool = False, base_path: str = "../data/processed") -> tuple:
    '''
    Load all data for a given country.

    Args:
        country (str): Country name.
        is_conflict (bool): Whether to read conflict data.
        base_path (str): Base path to the data directories.

    Returns:
        tuple: DataFrames of household, risk and damage, and optionally conflict data.
    '''
    paths = {
        'household': (Path(base_path) / "household_survey" / f"{country}.csv", 'csv'),
        'risk_and_damage': (Path(base_path) / "disaster_risk" / f"{country}.xlsx", 'xlsx'),
        'conflict': (Path(base_path) / "conflict" / f"{country}.xlsx", 'xlsx')
    }

    households = read_data_file(*paths['household'])
    risk_and_damage = read_data_file(*paths['risk_and_damage'])
    conflict = read_data_file(*paths['conflict']) if is_conflict else None

    return households, risk_and_damage, conflict
