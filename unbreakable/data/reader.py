"""This module contains functions to read data from files.
It reads two types of data: asset damage and household survey."""


import pandas as pd
import os
import numpy as np
import json


def read_asset_damage(country) -> None:
    '''Read asset damage data from an Excel file.'''
    if country == 'Saint Lucia':
        all_damage = pd.read_excel(
            f"../data/processed/asset_damage/{country}.xlsx", index_col=None, header=0)
    else:
        raise ValueError('Only `Saint Lucia` is supported.')

    return all_damage


def get_expected_loss_fraction(all_damage: pd.DataFrame, district: str, return_period: int) -> float:
    '''Get expected loss fraction for a district and return period.

    Args:
        all_damage (pd.DataFrame): Asset damage data for all districts.
        district (str): District name.
        return_period (int): Return period.

    Returns:
        float: Expected loss fraction.

    Raises:
        ValueError: If the return period is not in the data.
    '''
    # Check whether the return period is in the data
    return_periods = all_damage['rp'].unique()
    if return_period not in return_periods:
        raise ValueError(
            f'Return period {return_period} not found in the data.')

    # Get event damage for a specific district and return period
    event_damage = all_damage.loc[(all_damage['district'] == district) & (
        all_damage['rp'] == return_period), 'pml'].values[0]

    # Total here is residential and non-residential
    total_exposed_asset_stock = all_damage.loc[(all_damage['district'] == district) & (
        all_damage['rp'] == return_period), 'exposed_value'].values[0]
    expected_loss_fraction = event_damage / total_exposed_asset_stock
    return expected_loss_fraction


def get_total_exposed_asset_stock(all_damage: pd.DataFrame, district: str, return_period: int) -> float:
    '''Get total exposed asset stock for a district and return period.

    Args:
        all_damage (pd.DataFrame): Asset damage data for all districts.
        district (str): District name.
        return_period (int): Return period.

    Returns:
        float: Total exposed asset stock.
    '''
    # Total here is residential and non-residential
    total_exposed_asset_stock = all_damage.loc[(all_damage['district'] == district) & (
        all_damage['rp'] == return_period), 'exposed_value'].values[0]
    return total_exposed_asset_stock


def read_household_survey(country: str) -> pd.DataFrame:
    '''Reads household survey from a CSV file.

    Args:
        country (str): Country name.

    Returns:
        pd.DataFrame: Household survey data.

    Raises:
        ValueError: If the country is not `Saint Lucia`.
    '''
    if country == 'Saint Lucia':
        household_survey = pd.read_csv(
            f"../data/processed/household_survey/{country}.csv")
    else:
        raise ValueError('Only `Saint Lucia` is supported.')

    return household_survey
