import pandas as pd
import numpy as np


def duplicate_households(household_survey: pd.DataFrame, min_households: int) -> pd.DataFrame:
    '''Duplicates households if the number of households is less than `min_households` threshold.

    Args:
        household_survey (pd.DataFrame): Household survey data.
        min_households (int): Minimum number of households.

    Returns:
        pd.DataFrame: Household survey data with duplicated households.

    Raises:
        ValueError: If the total weights after duplication is not equal to the initial total weights.
    '''

    if len(household_survey) < min_households:
        print(
            f'Number of households = {len(household_survey)} is less than the threshold = {min_households}')

        initial_total_weights = household_survey['popwgt'].sum()

        # Save the original household id
        household_survey['hhid_original'] = household_survey[['hhid']]

        # Get random ids from the household data to be duplicated
        ids = np.random.choice(
            household_survey.index, min_households - len(household_survey), replace=True)
        n_duplicates = pd.Series(ids).value_counts() + 1
        duplicates = household_survey.loc[ids]

        # Adjust the weights of the duplicated households
        duplicates['popwgt'] = duplicates['popwgt'] / n_duplicates

        # Adjust the weights of the original households
        household_survey.loc[ids, 'popwgt'] = household_survey.loc[ids,
                                                                   'popwgt'] / n_duplicates

        # Combine the original and duplicated households
        household_survey = pd.concat(
            [household_survey, duplicates], ignore_index=True)

        # Check if the total weights after duplication is equal to the initial total weights
        # TODO: Allow for a small difference
        weights_after_duplication = household_survey['popwgt'].sum()
        if weights_after_duplication != initial_total_weights:
            raise ValueError(
                'Total weights after duplication is not equal to the initial total weights')

        household_survey.reset_index(drop=True, inplace=True)
        print(
            f'Number of households after duplication: {len(household_survey)}')
    else:
        return household_survey


def calculate_median_productivity(households: pd.DataFrame) -> pd.DataFrame:
    """Calculate the median productivity as a function of aeinc to k_house_ae, 
    
    where:
    - aeinc is the adult equivalent income (total)
    - k_house_ae is the domicile value divided by the ratio of household expenditure to adult equivalent expenditure (per capita), calculated as:
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
            f'Input DataFrame does not contain the required columns: {required_columns}')

    households['median_productivity'] = households['aeinc'] / households['k_house_ae']
    return households




def adjust_assets_and_expenditure(households: pd.DataFrame, total_asset_stock: float, poverty_line: float, indigence_line: float) -> pd.DataFrame:
    '''Adjust assets and expenditure of household to match data of asset damage file.

    There can be a mismatch between the data in the household survey and the of the asset damage.
    The latest was created independently.

    Args:
        households (pd.DataFrame): Household survey data.
        total_asset_stock (float): Total asset stock.
        poverty_line (float): Poverty line.
        indigence_line (float): Indigence line.

    Returns:
        pd.DataFrame: Household survey data with adjusted assets and expenditure.
    '''
    # TODO: Finish documentation
    # k_house_ae is domicile value divided by the ratio of household expenditure to adult equivalent expenditure (per capita) 
    # k_house_ae = data['domicile_value'] / (data['hhexp'] / data['aeexp'])
    # aeexp is adult equivalent expenditure (per capita)
    # aeexp_house is data['hhexp_house'] (annual rent) / data['hhsize_ae']
    included_variables = ['k_house_ae', 'aeexp', 'aeexp_house']

    # Save the initial values
    households['k_house_ae_original'] = households['k_house_ae']
    households['aeexp_original'] = households['aeexp']
    households['aeexp_house_original'] = households['aeexp_house']

    # Calculate the total asset in the survey
    total_asset_in_survey = households[[
        'popwgt', 'k_house_ae']].prod(axis=1).sum()
    households['total_asset_in_survey'] = total_asset_in_survey

    # Calculate the scaling factor and adjust the variables
    scaling_factor = total_asset_stock / total_asset_in_survey
    households[included_variables] *= scaling_factor
    households['poverty_line_adjusted'] = poverty_line * scaling_factor
    households['indigence_line_adjusted'] = indigence_line * scaling_factor

    return households


def calculate_household_probable_maximum_loss(households: pd.DataFrame, expected_loss_fraction: float) -> pd.DataFrame:
    '''Calculate probable maximum loss (PML) at the household level.
    
    PML here function of population weight, effective capital stock and expected loss fraction.

    Args:
        households (pd.DataFrame): Households.
        expected_loss_fraction (float): Expected loss fraction.

    Returns:
        pd.DataFrame: Household survey data with probable maximum loss.
    '''
    # k_house_ae is domicile value divided by the ratio of household expenditure to adult equivalent expenditure (per capita) 
    # k_house_ae = data['domicile_value'] / (data['hhexp'] / data['aeexp'])

    households['keff'] = households['k_house_ae'].copy()

    # TODO: Is it actually PML or something else?
    pml = households[['popwgt', 'keff']].prod(axis=1).sum() * expected_loss_fraction
    households['pml'] = pml
    return households

def estimate_savings(households: pd.DataFrame, saving_rate: float, estimate_savings_params: dict) -> pd.DataFrame:
    '''Estimate savings of households.

     We assume that savings are a product of expenditure and saving rate with Gaussian noise.

    Args:
        households (pd.DataFrame): Household survey data for a specific district.
        saving_rate (float): Saving rate.
        estimate_savings_params (dict): Parameters for estimating savings function.

    Returns:
        pd.DataFrame: Household survey data with estimated savings.
    '''
    # * Expenditure & savings information for Saint Lucia https://www.ceicdata.com/en/saint-lucia/lending-saving-and-deposit-rates-annual/lc-savings-rate

    # Savings are a product of expenditure and saving rate
    x = households.eval(f'aeexp*{saving_rate}')

    # Get the mean of the noise with uniform distribution
    mean_noise_low = estimate_savings_params['mean_noise_low']  # default 0
    mean_noise_high = estimate_savings_params['mean_noise_high']  # default 5

    if estimate_savings_params['mean_noise_distribution'] == 'uniform':
        loc = np.random.uniform(mean_noise_low, mean_noise_high)
    else:
        raise ValueError("Only uniform distribution is supported yet.")

    # Get the scale
    scale = estimate_savings_params['noise_scale']  # default 2.5
    size = households.shape[0]
    clip_min = estimate_savings_params['savings_clip_min']  # default 0.1
    clip_max = estimate_savings_params['savings_clip_max']  # default 1.0

    # Calculate savings with normal noise
    # !: aesav can go to 0 and above 1 because of the mean noise and loc
    # !: See `verification.ipynb` for more details
    if estimate_savings_params['noise_distribution'] == 'normal':
        households['aesav'] = x * \
            np.random.normal(loc, scale, size).round(
                2).clip(min=clip_min, max=clip_max)
    else:
        ValueError("Only normal distribution is supported yet.")

    return households


def assign_vulnerability(households: pd.DataFrame, is_vulnerability_random: bool, assign_vulnerability_params: dict) -> pd.DataFrame:
    '''Set vulnerability of households.

    Vulnerability can be random or based on `v_init` with uniform noise.

    Args:
        households (pd.DataFrame): Household survey data for a specific district.
        is_vulnerability_random (bool): If True, vulnerability is random.

    Returns:
        pd.DataFrame: Household survey data with assigned vulnerability.

    Raises:
        ValueError: If the distribution is not supported.
    '''

    # If vulnerability is random, then draw from the uniform distribution
    if is_vulnerability_random:
        # default 0.01
        low = assign_vulnerability_params['vulnerability_random_low']
        # default 0.90
        high = assign_vulnerability_params['vulnerability_random_high']
        if assign_vulnerability_params['vulnerability_random_distribution'] == 'uniform':
            households['v'] = np.random.uniform(low, high, households.shape[0])
        else:
            raise ValueError("Only uniform distribution is supported yet.")

    # If vulnerability is not random, use v_init as a starting point and add some noise
    # ?: What is the point of adding the noise to the v_init if we cap it anyhow
    else:
        # default 0.6
        low = assign_vulnerability_params['vulnerability_initial_low']
        # default 1.4
        high = assign_vulnerability_params['vulnerability_initial_high']
        # v - actual vulnerability
        # v_init - initial vulnerability
        if assign_vulnerability_params['vulnerability_initial_distribution'] == 'uniform':
            households['v'] = households['v_init'] * \
                np.random.uniform(low, high, households.shape[0])
        else:
            raise ValueError("Only uniform distribution is supported yet.")

        # default 0.95
        vulnerability_threshold = assign_vulnerability_params['vulnerability_initial_threshold']

        # If vulnerability turned out to be (drawn) is above the threshold, set it to the threshold
        households.loc[households['v'] >
                       vulnerability_threshold, 'v'] = vulnerability_threshold

        return households


def calculate_exposure(households: pd.DataFrame, poverty_bias: float, calculate_exposure_params: dict) -> pd.DataFrame:
    '''Calculate exposure of households.

    Exposure is a function of poverty bias, effective capital stock, 
    vulnerability and probable maximum loss.

    Args:
        households (pd.DataFrame): Household survey data for a specific district.
        poverty_bias (float): Poverty bias.
        calculate_exposure_params (dict): Parameters for calculating exposure function.

    Returns:
        pd.DataFrame: Household survey data with calculated exposure.
    '''
    pml = households['pml'].iloc[0]

    # Random value for poverty bias
    if poverty_bias == 'random':
        # default 0.5
        low = calculate_exposure_params['poverty_bias_random_low']
        # default 1.5
        high = calculate_exposure_params['poverty_bias_random_high']
        if calculate_exposure_params['poverty_bias_random_distribution'] == 'uniform':
            povbias = np.random.uniform(low, high)
        else:
            raise ValueError("Only uniform distribution is supported yet.")
    else:
        povbias = poverty_bias

    # Set poverty bias to 1 for all households
    households['poverty_bias'] = 1

    # Set poverty bias to povbias for poor households
    households.loc[households['is_poor'] == True, 'poverty_bias'] = povbias

    # DEFF: keff - effective capital stock
    delimiter = households[['keff', 'v', 'poverty_bias', 'popwgt']].prod(
        axis=1).sum()

    # ?: fa - fraction affected?
    fa0 = pml / delimiter

    households['fa'] = fa0 * households[['poverty_bias']]
    households.drop('poverty_bias', axis=1, inplace=True)
    return households


def identify_affected(households: pd.DataFrame, identify_affected_params: dict) -> tuple:
    '''Determines affected households.

    We assume that all households have the same probability of being affected, 
    but based on `fa` calculated in `calculate_exposure`.

    Args:
        households (pd.DataFrame): Household survey data for a specific district.
        determine_affected_params (dict): Parameters for determining affected households function.

    Returns:
        tuple: Household survey data with determined affected households and asset loss for each household.

    Raises:
        ValueError: If total asset is less than PML.
        ValueError: If no mask was found.
    '''
    # Get PML, it is the same for all households
    pml = households['pml'].iloc[0]

    # Allow for a relatively small error
    delta = pml * identify_affected_params['delta_pct']  # default 0.025

    # Check if total asset is less than PML
    total_asset = households[['keff', 'popwgt']].prod(axis=1).sum()
    if total_asset < pml:
        raise ValueError(
            'Total asset is less than PML.')

    low = identify_affected_params['low']  # default 0
    high = identify_affected_params['high']  # default 1

    # Generate multiple boolean masks at once
    num_masks = identify_affected_params['num_masks']  # default 2000
    masks = np.random.uniform(
        low, high, (num_masks, households.shape[0])) <= households['fa'].values

    # Compute total_asset_loss for each mask
    asset_losses = (
        masks * households[['keff', 'v', 'popwgt']].values.prod(axis=1)).sum(axis=1)

    # Find the first mask that yields a total_asset_loss within the desired range
    mask_index = np.where((asset_losses >= pml - delta) &
                          (asset_losses <= pml + delta))

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
        'keff', 'v', 'popwgt']].prod(axis=1).round(2)
    households['asset_loss'] = households['asset_loss'].fillna(0)

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
