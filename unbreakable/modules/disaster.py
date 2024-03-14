import pandas as pd


def get_region_damage(all_damage: pd.DataFrame, region: str, return_period: int) -> tuple[float, float, float]:
    '''
    Extracts total exposed stock, loss fraction and PML for a given region and return period.

    Args:
        all_damage (pd.DataFrame): DataFrame containing damage data.
        region (str): The region to filter the data.
        return_period (int): The return period to filter the data.

    Returns:
        tuple[float, float]: A tuple containing the total exposed stock, loss fraction and PML.
    '''

    # If the data has `rp` column, use it to filter the data
    if 'rp' in all_damage.columns:
        filtered_data = all_damage.loc[
            (all_damage['region'] == region) & (
                all_damage['rp'] == return_period),
            ['total_exposed_stock', 'loss_fraction', 'pml']
        ]
    # If the doesn't have `rp` column use only the `region` column to filter the data
    else:
        filtered_data = all_damage.loc[
            all_damage['region'] == region,
            ['total_exposed_stock', 'loss_fraction', 'pml']
        ]

    if filtered_data.empty:
        raise ValueError(
            "No data found for the specified region and/or return period.")

    return tuple(filtered_data.values[0])
