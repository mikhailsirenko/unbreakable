"""This module contains functions to read data from files.
It reads two types of data: asset damage and household survey."""


import pandas as pd
import os
import numpy as np
import json


def read_asset_damage(country: str) -> pd.DataFrame:
    '''Read asset damage data from an Excel file.

    Args:
        country (str): Country name.

    Returns:
        pd.DataFrame: Asset damage data.

    '''
    all_damage = pd.read_excel(
        f"../data/processed/asset_damage/{country}.xlsx", index_col=None, header=0)
    return all_damage


def read_household_survey(country: str) -> pd.DataFrame:
    '''Reads household survey from a CSV file.

    Args:
        country (str): Country name.

    Returns:
        pd.DataFrame: Household survey data.

    '''
    household_survey = pd.read_csv(
        f"../data/processed/household_survey/{country}.csv")
    return household_survey