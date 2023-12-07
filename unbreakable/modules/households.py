"""This module is the core of the model. 
It contains a set of functions that are used to calculate the impact of a disaster on households."""


import pandas as pd
import numpy as np


def match_assets_and_damage(households: pd.DataFrame, tot_exposed_asset: float, atol: bool) -> pd.DataFrame:
    '''Match assets and expenditure of to the asset damage data.

    There can be a mismatch between the asset stock in the household survey and the of the asset stock in the damage data.
    This function adjusts the asset stock and expenditure in the household survey to match the asset damage data.

    1). `k_house_ae` is domicile value divided by the ratio of household expenditure to adult equivalent expenditure (per capita)
    `k_house_ae` = domicile_value / (`hhexp` / `aeexp`)
    2). `aeexp` is adult equivalent expenditure (per capita)
    3). `aeexp_house` is `hhexp_house` (household annual rent) / `hhsize_ae`, 
    where `hhsize_ae` = `hhexp` / `aeexp`.

    Args:
        households (pd.DataFrame): Households.
        total_exposed_asset_stock (float): Total exposed asset stock from the damage data.
        indigence_line (float): Indigence line.
        atol (bool): Absolute tolerance for the comparison of the total exposed asset stock and the total asset stock in the survey.

    Returns:
        pd.DataFrame: Households with adjusted assets and expenditure.

    Raises:
        ValueError: If the total exposed asset stock is less than the total asset stock in the survey.
    '''

    # Get the total asset stock in the survey
    # In simple terms, k_house_ae is the price of a house
    tot_asset_surv = households[[
        'wgt', 'k_house_ae']].prod(axis=1).sum()

    # If the difference is small, return the original households (default atol = 100,000)
    if np.isclose(tot_exposed_asset, tot_asset_surv, atol=atol):
        return households
    else:
        # Save the initial values
        households['k_house_ae_orig'] = households['k_house_ae']
        households['aeexp_orig'] = households['aeexp']
        households['aeexp_house_orig'] = households['aeexp_house']

        # Calculate the total asset in the survey
        households['tot_asset_surv'] = tot_asset_surv

        # Calculate the scaling factor and adjust the variables
        scaling_factor = tot_exposed_asset / tot_asset_surv
        included_variables = ['k_house_ae', 'aeexp', 'aeexp_house']
        households[included_variables] *= scaling_factor
        poverty_line = households['povline'].iloc[0]
        households['poverty_line_adjusted'] = poverty_line * scaling_factor

        # Check the result of the adjustment
        tot_asset_surv_adjusted = households[['wgt', 'k_house_ae']].prod(
            axis=1).sum()

        if not np.isclose(tot_exposed_asset, tot_asset_surv_adjusted, atol=1e1):
            raise ValueError(
                'Total exposed asset stock is not equal to the total asset stock in the survey after adjustment.')

        return households


def calculate_pml(households: pd.DataFrame, expected_loss_frac: float) -> pd.DataFrame:
    '''Calculate the probable maximum loss (PML) of households in a district.

    PML here is a function of effective capital stock (`k_house_ae`) and expected loss fraction multiplied by the population weight of a household

    `k_house_ae` is domicile value divided by the ratio of household expenditure to adult equivalent expenditure (per capita)
    `k_house_ae` = domicile_value / (`hhexp` / `aeexp`)

    Args:
        households (pd.DataFrame): Households.
        expected_loss_frac (float): Expected loss fraction.

    Returns:
        pd.DataFrame: Households with calculated PML (`pml` column).
    '''
    households['keff'] = households['k_house_ae'].copy()
    district_pml = households[['wgt', 'keff']].prod(
        axis=1).sum() * expected_loss_frac

    # !: PML is the same for all households in a district
    households['pml'] = district_pml
    return households


def calculate_exposure(households: pd.DataFrame, poverty_bias: float, calc_exposure_params: dict) -> pd.DataFrame:
    '''Calculate the exposure of households.

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
