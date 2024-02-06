import pandas as pd


def get_stock_and_loss_fraction(all_damage: pd.DataFrame, district: str, return_period: int) -> tuple[float, float]:
    '''
    Extracts total exposed stock and loss fraction for a given district and return period.

    Args:
        all_damage (pd.DataFrame): DataFrame containing damage data.
        district (str): The district to filter the data.
        return_period (int): The return period to filter the data.

    Returns:
        tuple[float, float]: A tuple containing the total exposed stock and loss fraction.
    '''
    filtered_data = all_damage.loc[
        (all_damage['district'] == district) & (
            all_damage['rp'] == return_period),
        ['total_exposed_stock', 'loss_fraction', 'pml']
    ]

    if filtered_data.empty:
        raise ValueError(
            "No data found for the specified district and return period.")

    return tuple(filtered_data.values[0])
