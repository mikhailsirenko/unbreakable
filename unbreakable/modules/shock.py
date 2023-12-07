"""This module """

import pandas as pd


def get_expected_loss_frac(all_damage: pd.DataFrame, district: str, return_period: int) -> float:
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


def get_tot_exposed_asset_stock(all_damage: pd.DataFrame, district: str, return_period: int) -> float:
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
