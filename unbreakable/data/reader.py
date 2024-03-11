import pandas as pd
import logging
from pathlib import Path


def read_risk_and_damage(country: str, base_path: str = "../data/processed/disaster_risk", file_extension: str = "xlsx") -> pd.DataFrame:
    '''
    Read disaster risk assessment data from a file.

    Args:
        country (str): Country name.
        base_path (str): Base path to the disaster risk assessment data directory.
        file_extension (str): File extension (default is 'xlsx').

    Returns:
        pd.DataFrame: Disaster risk assessment data.
    '''
    file_path = Path(base_path) / f"{country}.{file_extension}"
    try:
        data = pd.read_excel(
            file_path, index_col=None, header=0) if file_extension == 'xlsx' else pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(
            f"Error reading disaster risk assessment data from {file_path}: {e}")
        raise


def read_household_survey(country: str, base_path: str = "../data/processed/household_survey", file_extension: str = "csv") -> pd.DataFrame:
    '''
    Reads household survey data from a file.

    Args:
        country (str): Country name.
        base_path (str): Base path to the household survey data directory.
        file_extension (str): File extension (default is 'csv').

    Returns:
        pd.DataFrame: Household survey data.
    '''
    file_path = Path(base_path) / f"{country}.{file_extension}"
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(
            f"Error reading household survey data from {file_path}: {e}")
        raise


def read_conflict_data(country: str, base_path: str = "../data/processed/conflict/", file_extension: str = "csv") -> pd.DataFrame:
    '''
    Reads conflict data from a file.

    Args:
        country (str): Country name.
        base_path (str): Base path to the conflict data directory.
        file_extension (str): File extension (default is 'csv').

    Returns:
        pd.DataFrame: Conflict data.
    '''
    file_path = Path(base_path) / f"{country}.{file_extension}"
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(
            f"Error reading conflict data from {file_path}: {e}")
        raise


def load_data(country):
    '''Load all data for a given country.'''
    all_households = read_household_survey(country)
    risk_and_damage = read_risk_and_damage(country)
    return all_households, risk_and_damage
