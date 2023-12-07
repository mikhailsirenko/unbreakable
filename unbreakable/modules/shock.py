"""This module provides functions for calculating expected loss fraction and total exposed asset stock for a given district and return period.

The main function in this module is `get_loss_fraction_and_exposed_stock`, which takes asset damage data for all districts, a district name, and a return period as input. It returns the expected loss fraction and total exposed asset stock for the specified district and return period.

Example usage:
    import pandas as pd
    from unbreakable.modules.shock import get_loss_fraction_and_exposed_stock

    # Load asset damage data
    all_damage = pd.read_csv('asset_damage.csv')

    # Calculate expected loss fraction and total exposed asset stock for district 'ABC' and return period 10
    loss_fraction, exposed_stock = get_loss_fraction_and_exposed_stock(all_damage, 'ABC', 10)

    print(f'Expected loss fraction: {loss_fraction}')
    print(f'Total exposed asset stock: {exposed_stock}')
"""

import pandas as pd


def get_loss_fraction_and_exposed_stock(all_damage: pd.DataFrame, district: str, return_period: int) -> tuple[float, float]:
    '''Get expected loss fraction and total exposed asset stock for a district and return period.

    Args:
        all_damage (pd.DataFrame): Asset damage data for all districts.
            Required columns: 'district', 'rp', 'pml', 'exposed_value'
        district (str): District name.
        return_period (int): Return period.

    Returns:
        tuple[float, float]: Expected loss fraction and total exposed asset stock.

    Raises:
        ValueError: If the district and return period are not in the data.
    '''
    # Check whether the district and return period are in the data
    filtered_data = all_damage.loc[(all_damage['district'] == district) & (
        all_damage['rp'] == return_period)]

    if filtered_data.empty:
        raise ValueError(
            f'District {district} and return period {return_period} not found in the data.')

    # Get event damage and total exposed asset stock
    event_damage = filtered_data.at[filtered_data.index[0], 'pml']
    total_exposed_asset_stock = filtered_data.at[filtered_data.index[0], 'exposed_value']

    # Calculate expected loss fraction
    expected_loss_fraction = event_damage / total_exposed_asset_stock

    return expected_loss_fraction, total_exposed_asset_stock
