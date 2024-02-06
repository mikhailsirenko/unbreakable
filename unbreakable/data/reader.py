"""This module provides functions for reading and processing data related to disaster impact analysis and recovery planning.

It includes two primary functions:
1. `read_asset_damage`: Reads asset damage data from a specified file location. This data is essential for assessing the physical and monetary losses incurred by assets in different districts due to disasters.
2. `read_household_survey`: Reads household survey data from a specified file location. This data provides insights into the demographics, economic conditions, and vulnerabilities of households, which are crucial for effective disaster response and recovery planning.

Both functions are designed to handle common file formats such as CSV and Excel, and include error handling for robust data processing.

Example usage:
    import pandas as pd
    from your_module_name import read_asset_damage, read_household_survey

    # Load asset damage data
    asset_damage = read_asset_damage('CountryName')

    # Load household survey data
    household_survey = read_household_survey('CountryName')

    # Now asset_damage and household_survey are pandas DataFrames containing the respective data.
"""

import pandas as pd
import logging
from pathlib import Path


def read_disaster_risk_data(country: str, base_path: str = "../data/processed/disaster_risk_assessment", file_extension: str = "xlsx") -> pd.DataFrame:
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
