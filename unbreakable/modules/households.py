"""This module is the core of the model. 
It contains a set of functions that are used to calculate the impact of a disaster on households."""


import pandas as pd
import numpy as np


# What happens here is that we
def calculate_exposure(households: pd.DataFrame, poverty_bias: float, calc_exposure_params: dict) -> pd.DataFrame:
    '''Calculate exposure of households to the disaster.

    Exposure is a function of poverty bias, effective capital stock, 
    vulnerability and probable maximum loss.

    Args:
        households (pd.DataFrame): Households.
        poverty_bias (float): Poverty bias.
        calc_exposure_params (dict): Parameters for calculating exposure function.

    Returns:
        pd.DataFrame: Households with calculated exposure (`fa` column).
    '''
    district_pml = households['pml'].iloc[0]

    # Random value for poverty bias
    if poverty_bias == 'random':
        if calc_exposure_params['pov_bias_rnd_distr'] == 'uniform':
            # default 0.5
            low = calc_exposure_params['pov_bias_rnd_low']
            # default 1.5
            high = calc_exposure_params['pov_bias_rnd_high']
            povbias = np.random.uniform(low, high)
        else:
            raise ValueError("Only uniform distribution is supported yet.")
    else:
        povbias = poverty_bias

    # Set poverty bias to 1 for all households
    households['poverty_bias'] = 1

    # Set poverty bias to povbias for poor households
    households.loc[households['is_poor'] == True, 'poverty_bias'] = povbias

    # !: Get rid of keff
    households['keff'] = households['k_house_ae'].copy()

    delimiter = households[['keff', 'v', 'poverty_bias', 'wgt']].prod(
        axis=1).sum()

    fa0 = district_pml / delimiter
    households['fa'] = fa0 * households[['poverty_bias']]
    households.drop('poverty_bias', axis=1, inplace=True)
    return households


def identify_affected(households: pd.DataFrame, ident_affected_params: dict) -> tuple:
    '''Determines affected households.

    We assume that all households have the same probability of being affected, 
    but based on `fa` value calculated in `calculate_exposure`.

    Args:
        households (pd.DataFrame): Households.
        ident_affected_params (dict): Parameters for determining affected households function.

    Returns:
        tuple: Households with `is_affected` and `asset_loss` columns.

    Raises:
        ValueError: If no mask was found.
    '''
    # Get PML, it is the same for all households
    district_pml = households['pml'].iloc[0]

    # Allow for a relatively small error
    delta = district_pml * ident_affected_params['delta_pct']  # default 0.025

    # Check if total asset is less than PML
    tot_asset_stock = households[['keff', 'wgt']].prod(axis=1).sum()
    if tot_asset_stock < district_pml:
        raise ValueError(
            'Total asset stock is less than PML.')

    low = ident_affected_params['low']  # default 0
    high = ident_affected_params['high']  # default 1

    # Generate multiple boolean masks at once
    num_masks = ident_affected_params['num_masks']  # default 2000
    masks = np.random.uniform(
        low, high, (num_masks, households.shape[0])) <= households['fa'].values

    # Compute total_asset_loss for each mask
    asset_losses = (
        masks * households[['keff', 'v', 'wgt']].values.prod(axis=1)).sum(axis=1)

    # Find the first mask that yields a total_asset_loss within the desired range
    mask_index = np.where((asset_losses >= district_pml - delta) &
                          (asset_losses <= district_pml + delta))

    # Raise an error if no mask was found
    if mask_index is None:
        raise ValueError(
            f'Cannot find affected households in {num_masks} iterations.')
    else:
        try:
            # Select the first mask that satisfies the condition
            mask_index = mask_index[0][0]
        except:
            print('mask_index: ', mask_index)

    chosen_mask = masks[mask_index]

    # Raise an error if no mask was found
    if len(chosen_mask) == 0:
        raise ValueError(
            f'Cannot find affected households in {num_masks} iterations.')

    # Assign the chosen mask to the 'is_affected' column of the DataFrame
    households['is_affected'] = chosen_mask

    # Save the asset loss for each household
    households['asset_loss'] = households.loc[households['is_affected'], [
        'keff', 'v', 'wgt']].prod(axis=1)
    households['asset_loss'] = households['asset_loss'].fillna(0)

    # Check whether the total asset loss is within the desired range
    tot_asset_loss = households['asset_loss'].sum()
    if (tot_asset_loss < district_pml - delta) or (tot_asset_loss > district_pml + delta):
        raise ValueError(
            f'Total asset loss ({tot_asset_loss}) is not within the desired range.')

    return households
