import pandas as pd
import numpy as np


def calculate_median_productivity(households: pd.DataFrame) -> pd.DataFrame:
    """Calculate the median productivity as a function of aeinc to k_house_ae, 

    where:
    - aeinc is the adult equivalent income (total)
    - k_house_ae is the domicile value divided by 
    the ratio of household expenditure to adult equivalent expenditure (per capita), calculated as:
      k_house_ae = domicile_value / (hhexp / aeexp)

    Args:
        households (pd.DataFrame): A households DataFrame containing the columns 'aeinc' and 'k_house_ae'.

    Returns:
        float: Median productivity.

    Raises:
        ValueError: If the input DataFrame does not contain the required columns.

    Note:
        The function ignores NaN values in the calculation of the median.
    """
    if not all(column in households.columns for column in ['aeinc', 'k_house_ae']):
        raise ValueError(
            f'Input DataFrame does not contain the required columns: ["aeinc", "k_house_ae"]')

    households['median_productivity'] = households['aeinc'] / \
        households['k_house_ae']
    return households


def duplicate_households(households: pd.DataFrame, min_representative_households: int) -> pd.DataFrame:
    '''Duplicates households if the number of households is less than `min_households` threshold.

    Args:
        households (pd.DataFrame): Households of a specific district.
        min_representative_households (int): Minimum number of households for the district sample to be representative.

    Returns:
        pd.DataFrame: Duplicated households of a specific district.

    Raises:
        ValueError: If the total weights after duplication is not equal to the initial total weights.
    '''

    if len(households) < min_representative_households:
        # Save the initial total weights
        initial_total_weights = households['popwgt'].sum()

        # Save the original household id
        households['hhid_original'] = households[['hhid']]

        # Sample households with replacement
        delta = min_representative_households - len(households)

        # To avoid sampling one household
        if delta == 1:
            delta = 2

        sample = households.sample(n=delta, replace=True)

        # Keep how many duplicates by index
        duplicates = sample.index.value_counts()

        # Combine the original and the sampled households
        duplicated_households = pd.concat(
            [households, sample], axis=0, sort=False)

        # Iterate over the duplicated households and update the weights
        for household_id in duplicates.index:
            # Get original weight
            original_weight = households.loc[household_id, 'popwgt']

            # Count how many rows are there (could be one as well)
            n_duplicates = duplicates[household_id]

            # Update the weight
            duplicated_households.loc[household_id,
                                      'popwgt'] = original_weight / (n_duplicates + 1)

            # Get the new weight
            weights_after_duplication = duplicated_households.loc[household_id, 'popwgt'].sum(
            )

            # Check if it is close to the original weight
            if not np.isclose(weights_after_duplication, original_weight, atol=1e-6):
                raise ValueError(
                    'Total weights after duplication is not equal to the initial total weights')

        if not np.isclose(duplicated_households['popwgt'].sum(), initial_total_weights, atol=1e-6):
            raise ValueError(
                'Total weights after duplication is not equal to the initial total weights')

        return duplicated_households.reset_index(drop=True)
    else:
        return households


def match_assets_and_expenditure(households: pd.DataFrame, tot_exposed_asset: float, poverty_line: float, indigence_line: float, atol: bool) -> pd.DataFrame:
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
        poverty_line (float): Poverty line.
        indigence_line (float): Indigence line.
        atol (bool): Absolute tolerance for the comparison of the total exposed asset stock and the total asset stock in the survey.

    Returns:
        pd.DataFrame: Households with adjusted assets and expenditure.

    Raises:
        ValueError: If the total exposed asset stock is less than the total asset stock in the survey.
    '''

    tot_asset_surv = households[[
        'popwgt', 'k_house_ae']].prod(axis=1).sum()
    # If the difference is small, return the original households (default rtol = 100,000)
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
        households['poverty_line_adjusted'] = poverty_line * scaling_factor
        households['indigence_line_adjusted'] = indigence_line * scaling_factor

        # Check the result of the adjustment
        tot_asset_surv_adjusted = households[['popwgt', 'k_house_ae']].prod(
            axis=1).sum()
        
        if not np.isclose(tot_exposed_asset, tot_asset_surv_adjusted, atol=1e1):
            raise ValueError(
                'Total exposed asset stock is not equal to the total asset stock in the survey after adjustment')

        return households


def calculate_district_pml(households: pd.DataFrame, expected_loss_frac: float) -> pd.DataFrame:
    '''Calculate probable maximum loss (PML) of households in a district.

    PML here function of effective capital stock (`k_house_ae`) and expected loss fraction (of course, multiplied by the population weight of a household)

    `k_house_ae` is domicile value divided by the ratio of household expenditure to adult equivalent expenditure (per capita)
    `k_house_ae` = domicile_value / (`hhexp` / `aeexp`)

    Args:
        households (pd.DataFrame): Households.
        expected_loss_frac (float): Expected loss fraction.

    Returns:
        pd.DataFrame: Households with a new column `pml` for probable maximum loss.
    '''
    households['keff'] = households['k_house_ae'].copy()
    district_pml = households[['popwgt', 'keff']].prod(
        axis=1).sum() * expected_loss_frac

    # PML is the same for all households
    households['district_pml'] = district_pml
    return households


def estimate_savings(households: pd.DataFrame, saving_rate: float, est_sav_params: dict) -> pd.DataFrame:
    '''Estimate savings of households.

     We assume that savings are a product of expenditure and saving rate with Gaussian noise.

    Args:
        households (pd.DataFrame): Households.
        saving_rate (float): Saving rate.
        est_sav_params (dict): Parameters for estimating savings function.

    Returns:
        pd.DataFrame: Households with estimated savings column `aesav`.

    Raises:
        ValueError: If the distribution of the noise is not supported.
    '''
    # * Expenditure & savings information for Saint Lucia https://www.ceicdata.com/en/saint-lucia/lending-saving-and-deposit-rates-annual/lc-savings-rate

    # Savings are a product of expenditure and saving rate
    x = households.eval(f'aeexp*{saving_rate}')  # default 0.02385 ir 2.385%

    # Get the mean of the noise with uniform distribution
    mean_noise_low = est_sav_params['mean_noise_low']  # default 0
    mean_noise_high = est_sav_params['mean_noise_high']  # default 5

    if est_sav_params['mean_noise_distr'] == 'uniform':
        loc = np.random.uniform(mean_noise_low, mean_noise_high)
    else:
        raise ValueError("Only uniform distribution is supported yet.")

    # Get the scale
    scale = est_sav_params['noise_scale']  # default 2.5
    size = households.shape[0]
    clip_min = est_sav_params['sav_clip_min']  # default 0.1
    clip_max = est_sav_params['sav_clip_max']  # default 1.0

    # Calculate savings with normal noise
    # !: `aesav` can go to 0 and above 1 because of the mean noise and loc
    # !: See `verification.ipynb` for more details
    if est_sav_params['noise_distr'] == 'normal':
        households['aesav'] = x * \
            np.random.normal(loc, scale, size).round(
                2).clip(min=clip_min, max=clip_max)
    else:
        ValueError("Only normal distribution is supported yet.")

    return households


def assign_vulnerability(households: pd.DataFrame, is_vuln_random: bool, assign_vuln_params: dict) -> pd.DataFrame:
    '''Assign vulnerability to each household.

    Vulnerability can be random or based on `v_init` with uniform noise.

    Args:
        households (pd.DataFrame): Households.
        is_vulnerability_random (bool): If True, vulnerability is random.

    Returns:
        pd.DataFrame: Households with assigned vulnerability `v`.

    Raises:
        ValueError: If the distribution is not supported.
    '''

    # If vulnerability is random, then draw from the uniform distribution
    if is_vuln_random:
        # default 0.01
        low = assign_vuln_params['vuln_rnd_low']
        # default 0.90
        high = assign_vuln_params['vuln_rnd_high']
        if assign_vuln_params['vuln_rnd_distr'] == 'uniform':
            households['v'] = np.random.uniform(low, high, households.shape[0])
        else:
            raise ValueError("Only uniform distribution is supported yet.")

    # If vulnerability is not random, use v_init as a starting point and add some noise
    # ?: What is the point of adding the noise to the v_init if we cap it anyhow
    else:
        # default 0.6
        low = assign_vuln_params['vuln_init_low']
        # default 1.4
        high = assign_vuln_params['vuln_init_high']
        # v - actual vulnerability
        # v_init - initial vulnerability
        if assign_vuln_params['vuln_init_distr'] == 'uniform':
            households['v'] = households['v_init'] * \
                np.random.uniform(low, high, households.shape[0])
        else:
            raise ValueError("Only uniform distribution is supported yet.")

        # default 0.95
        vulnerability_threshold = assign_vuln_params['vuln_init_thresh']

        # If vulnerability turned out to be (drawn) is above the threshold, set it to the threshold
        households.loc[households['v'] >
                       vulnerability_threshold, 'v'] = vulnerability_threshold

        return households


def calculate_exposure(households: pd.DataFrame, poverty_bias: float, calc_exposure_params: dict) -> pd.DataFrame:
    '''Calculate exposure of households.

    Exposure is a function of poverty bias, effective capital stock, 
    vulnerability and probable maximum loss.

    Args:
        households (pd.DataFrame): Households.
        poverty_bias (float): Poverty bias.
        calc_exposure_params (dict): Parameters for calculating exposure function.

    Returns:
        pd.DataFrame: Households with calculated exposure (`fa` column).
    '''
    district_pml = households['district_pml'].iloc[0]

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

    delimiter = households[['keff', 'v', 'poverty_bias', 'popwgt']].prod(
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
    district_pml = households['district_pml'].iloc[0]

    # Allow for a relatively small error
    delta = district_pml * ident_affected_params['delta_pct']  # default 0.025

    # Check if total asset is less than PML
    tot_asset_stock = households[['keff', 'popwgt']].prod(axis=1).sum()
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
        masks * households[['keff', 'v', 'popwgt']].values.prod(axis=1)).sum(axis=1)

    # Find the first mask that yields a total_asset_loss within the desired range
    mask_index = np.where((asset_losses >= district_pml - delta) &
                          (asset_losses <= district_pml + delta))

    # Raise an error if no mask was found
    if mask_index is None:
        raise ValueError(
            f'Cannot find affected households in {num_masks} iterations.')
    else:
        try:
            mask_index = mask_index[0][0]
        except:
            print('mask_index: ', mask_index)

    chosen_mask = masks[mask_index]

    # Assign the chosen mask to the 'is_affected' column of the DataFrame
    households['is_affected'] = chosen_mask

    # Save the asset loss for each household
    households['asset_loss'] = households.loc[households['is_affected'], [
        'keff', 'v', 'popwgt']].prod(axis=1)
    households['asset_loss'] = households['asset_loss'].fillna(0)

    # Check whether the total asset loss is within the desired range
    tot_asset_loss = households['asset_loss'].sum()
    if (tot_asset_loss < district_pml - delta) or (tot_asset_loss > district_pml + delta):
        raise ValueError(
            f'Total asset loss ({tot_asset_loss}) is not within the desired range.') 

    return households


def apply_policy(households: pd.DataFrame, my_policy: str) -> pd.DataFrame:
    '''Apply a policy to a specific target group.

    Args:
        households (pd.DataFrame): Households.
        my_policy (str): Policy to apply. Format: <target_group>+<top_up>. Example: "poor+100".
    Returns:
        pd.DataFrame: Households with applied policy.
    '''
    # * Note that we adjusted the poverty line in `adjust_assets_and_expenditure`
    poverty_line_adjusted = households['poverty_line_adjusted'].iloc[0]

    target_group, top_up = my_policy.split('+')
    top_up = float(top_up)

    if target_group == 'all':
        beneficiaries = households['is_affected'] == True

    elif target_group == 'poor':
        beneficiaries = (households['is_affected'] == True) & (
            households['is_poor'] == True)

    elif target_group == 'poor_near_poor1.25':
        poor_affected = (households['is_affected'] == True) & (
            households['is_poor'] == True)
        near_poor_affected = (households['is_affected'] == True) & (
            households['is_poor'] == False) & (households['aeexp'] < 1.25 * poverty_line_adjusted)
        beneficiaries = poor_affected | near_poor_affected

    elif target_group == 'poor_near_poor2.0':
        poor_affected = (households['is_affected'] == True) & (
            households['is_poor'] == True)
        near_poor_affected = (households['is_affected'] == True) & (
            households['is_poor'] == False) & (households['aeexp'] < 2 * poverty_line_adjusted)
        beneficiaries = poor_affected | near_poor_affected

    households.loc[beneficiaries,
                   'aesav'] += households.loc[beneficiaries].eval('keff*v') * top_up / 100

    return households
