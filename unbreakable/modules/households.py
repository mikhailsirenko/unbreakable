import pandas as pd
import numpy as np


def estimate_impact(households: pd.DataFrame, region_pml: float, pov_bias: float, calc_exposure_params: dict) -> pd.DataFrame:
    # Random value for poverty bias
    if pov_bias == 'random':
        if calc_exposure_params['distr'] == 'uniform':
            # default 0.5
            low = calc_exposure_params['low']
            # default 1.5
            high = calc_exposure_params['high']
            povbias = np.random.uniform(low, high)
        else:
            raise ValueError("Only uniform distribution is supported yet.")
    else:
        povbias = pov_bias

    # Set poverty bias to 1 for all households
    households['pov_bias'] = 1

    # Set poverty bias to povbias for poor households
    households.loc[households['is_poor'] == True, 'pov_bias'] = povbias

    # !: If keff is just k_house, why not just use k_house?
    households['keff'] = households['k_house'].copy()

    # Calculate the exposure of each household
    normalization_factor = households[['keff', 'v', 'pov_bias', 'wgt']].prod(
        axis=1).sum()

    # Distribute the impact of the disaster across households
    households['fa'] = (region_pml / normalization_factor) * \
        households[['pov_bias']]

    return households


def identify_affected(households: pd.DataFrame, region_pml: float, ident_affected_params: dict, random_seed: int) -> pd.DataFrame:
    '''Determines affected households.

    We assume that all households have the same chance of being affected, 
    but based on `fa` value calculated in `estimate_impact`.

    Args:
        households (pd.DataFrame): Households.
        district_pml (float): Region PML.
        ident_affected_params (dict): Parameters for determining affected households function.

    Returns:
        pd.DataFrame: Households with `is_affected` and `asset_loss` columns.

    Raises:
        ValueError: If no mask was found.
    '''
    # BUG: For whatever reason fixing random seed in the model.py doesn't work here
    np.random.seed(random_seed)

    # Allow for a relatively small error
    delta = region_pml * ident_affected_params['delta_pct']  # default 0.025

    # Check if total asset is less than PML
    tot_asset_stock = households[['keff', 'wgt']].prod(axis=1).sum()
    if tot_asset_stock < region_pml:
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
    mask_index = np.where((asset_losses >= region_pml - delta) &
                          (asset_losses <= region_pml + delta))

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
    if (tot_asset_loss < region_pml - delta) or (tot_asset_loss > region_pml + delta):
        raise ValueError(
            f'Total asset loss ({tot_asset_loss}) is not within the desired range.')

    return households


def calculate_welfare(country_households: pd.DataFrame, cons_util: float) -> float:
    # TODO: Make sure that the function name and what is returns is correct
    # TODO: Add a docstring
    welfare = (np.sum(country_households['exp'] * country_households['wgt']) / np.sum(
        country_households['wgt'])) ** (-cons_util)
    return welfare
