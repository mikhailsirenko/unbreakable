import pandas as pd
import os
import numpy as np
import json


def read_asset_damage(country) -> None:
    '''Read asset damage for all districts from a XLSX file and load it into the memory.'''
    if country == 'Saint Lucia':
        all_damage = pd.read_excel(
            f"../data/processed/asset_damage/{country}/{country}.xlsx", index_col=None, header=0)
    else:
        raise ValueError('Only `Saint Lucia` is supported.')

    return all_damage


def get_asset_damage(all_damage: pd.DataFrame, scale: str, district: str, return_period: int, print_statistics: bool) -> tuple:
    '''Get asset damage for a specific district.

    Args:
        all_damage (pd.DataFrame): Asset damage data for all districts.
        scale (str): Scale of the analysis. Only `district` is supported.
        district (str): District name.
        return_period (int): Return period.
        print_statistics (bool): Print the statistics.

    Returns:
        tuple: Event damage, total asset stock, expected loss fraction.

    Raises:
        ValueError: If the scale is not `district`.   
        ValueError: If the expected loss fraction is greater than 1.
    '''
    if scale == 'district':
        event_damage = all_damage.loc[(all_damage[scale] == district) & (
            all_damage['rp'] == return_period), 'pml'].values[0]  # PML
        total_asset_stock = all_damage.loc[(all_damage[scale] == district) & (
            all_damage['rp'] == return_period), 'exposed_value'].values[0]  # Exposed value

    else:
        raise ValueError(
            'Only `district` scale is supported.')

    event_damage = event_damage
    total_asset_stock = total_asset_stock
    expected_loss_fraction = event_damage / total_asset_stock

    if expected_loss_fraction > 1:
        raise ValueError(
            'Expected loss fraction is greater than 1. Check the data.')

    if print_statistics:
        print('Event damage = ' + str('{:,}'.format(round(event_damage))))
        print('Total asset stock = ' +
              str('{:,}'.format(round(total_asset_stock))))
        print('Expected loss fraction = ' +
              str(np.round(expected_loss_fraction, 3)))

    return event_damage, total_asset_stock, expected_loss_fraction


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
            f"../data/processed/household_survey/{country}/{country}.csv")
    else:
        raise ValueError('Only `Saint Lucia` is supported.')

    return household_survey