import pandas as pd
import numpy as np
import random
from utils.reader import *
from optimize.optimizer import *
from tests.tester import *
from utils.writer import *
import pickle
import time

# TODO: Move RaiseValueError to tests


def initialize_model(country: str, scale: str, min_households: int) -> tuple:
    '''Initialize the model.

    Read household survey and asset damage files.

    Args:
        country (str): Country name.
        scale (str): Scale of the model. Available options: district.

    Returns:
        tuple: Household survey and asset damage files.
    '''    
    test_country(country)
    test_scale(scale)

    # Read household survey and asset damage files
    household_survey = read_household_survey(country)

    # Duplicate households to have at least `min_households` households
    household_survey = duplicate_households(household_survey, min_households)
    all_damage = read_asset_damage(country)

    return household_survey, all_damage


def run_model(**kwargs):
    # Case study parameters
    country = kwargs['country']
    scale = kwargs['scale']
    districts = kwargs['districts']
    min_households = kwargs['min_households']

    # Read household survey and asset damage files
    household_survey, all_damage = initialize_model(
        country, scale, min_households)

    # Case study constants
    return_period = kwargs['return_period']
    poverty_line = kwargs['poverty_line']
    indigence_line = kwargs['indigence_line']
    saving_rate = kwargs['saving_rate']
    is_vulnerability_random = kwargs['is_vulnerability_random']
    x_max = kwargs['x_max']  # number of years in optimization algorithm

    # Model constants
    assign_savings_params = kwargs['assign_savings_params']
    set_vulnerability_params = kwargs['set_vulnerability_params']
    calculate_exposure_params = kwargs['calculate_exposure_params']
    determine_affected_params = kwargs['determine_affected_params']

    # Uncertainties
    poverty_bias = kwargs['poverty_bias']
    consumption_utility = kwargs['consumption_utility']
    discount_rate = kwargs['discount_rate']
    optimization_timestep = kwargs['optimization_timestep']
    income_and_expenditure_growth = kwargs['income_and_expenditure_growth']

    # Policy levers
    try:
        top_up = kwargs['top_up']
        target_group = kwargs['target_group']
    except:
        top_up = 0
        target_group = 'all'

    # Outcomes
    # Store outcomes in a dictionary, where key is a district and value is a dictionary of outcomes
    outcomes = {}

    # Print statistics for debugging
    print_statistics = kwargs['print_statistics']

    # Fix random seed for reproducibility
    random_seed = kwargs['random_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)

    # PIPELINE:
    # 1. Load all damage
    # 2. Load household survey
    # 3. Select district
    # 4. Get event damage, total asset stock, expected loss fraction
    # 5. Calculate average productivity
    # 6. Adjust assets and expenditure
    # 7. Calculate PML
    # 8. Assign savings
    # 9. Set vulnerability
    # 10. Calculate exposure
    # 11. Determine affected
    # 12. Apply individual policy
    # 13. Run optimization
    # 14. Integrate wellbeing
    # 15. Prepare outcomes
    # 16. Get outcomes

    for district in districts:
        # Read household survey and asset damage files for a specific district
        event_damage, total_asset_stock, expected_loss_fraction = get_asset_damage(
            all_damage, scale, district, return_period, print_statistics)
        households = select_district(household_survey, district)
        average_productivity = calculate_average_productivity(
            households, print_statistics)

        # Model the impact of a disaster on households
        households = (adjust_assets_and_expenditure(households, total_asset_stock, poverty_line, indigence_line, print_statistics)
                      .pipe(calculate_pml, expected_loss_fraction, print_statistics)
                      .pipe(assign_savings, saving_rate, assign_savings_params)
                      .pipe(set_vulnerability, is_vulnerability_random, set_vulnerability_params)
                      .pipe(calculate_exposure, poverty_bias, calculate_exposure_params, print_statistics)
                      .pipe(determine_affected, determine_affected_params))

        # Apply a policy
        households, affected_households = apply_individual_policy(
            households, top_up, target_group, poverty_line)

        # Calculate the impact and recovery
        affected_households = (run_optimization(affected_households, consumption_utility, discount_rate, average_productivity, optimization_timestep)
                               .pipe(integrate_wellbeing, consumption_utility, discount_rate, income_and_expenditure_growth, average_productivity, poverty_line, x_max))

        # Get outcomes
        households = prepare_outcomes(households, affected_households)
        array_outcomes = np.array(list(get_outcomes(
            households, event_damage, total_asset_stock, expected_loss_fraction, average_productivity, x_max).values()))
        outcomes[district] = array_outcomes

    return outcomes


def select_district(household_survey: pd.DataFrame, district: str) -> pd.DataFrame:
    '''Select households for a specific district.'''
    return household_survey[household_survey['district'] == district].copy()


def assign_savings(households: pd.DataFrame, saving_rate: float, assign_savings_params: dict) -> pd.DataFrame:
    '''Assign savings to households.

     We assume that savings are a product of expenditure and saving rate with Gaussian noise.

    Args:
        households (pd.DataFrame): Household survey data for a specific district.
        saving_rate (float): Saving rate.
        assign_savings_params (dict): Parameters for assigning savings function.

    Returns:
        pd.DataFrame: Household survey data with assigned savings.
    '''
    # * Expenditure & savings information for Saint Lucia https://www.ceicdata.com/en/saint-lucia/lending-saving-and-deposit-rates-annual/lc-savings-rate

    # Savings are a product of expenditure and saving rate
    x = households.eval(f'aeexp*{saving_rate}')

    # Get the mean of the noise with uniform distribution
    mean_noise_low = assign_savings_params['mean_noise_low']  # default 0
    mean_noise_high = assign_savings_params['mean_noise_high']  # default 5

    if assign_savings_params['mean_noise_distribution'] == 'uniform':
        loc = np.random.uniform(mean_noise_low, mean_noise_high)
    else:
        raise ValueError("Only uniform distribution is supported yet.")

    # Get the scale
    scale = assign_savings_params['noise_scale']  # default 2.5
    size = households.shape[0]
    clip_min = assign_savings_params['savings_clip_min']  # default 0.1
    clip_max = assign_savings_params['savings_clip_max']  # default 1.0

    # Calculate savings with normal noise
    # !: aesav can go to 0 and above 1 because of the mean noise and loc
    # !: See `verification.ipynb` for more details
    if assign_savings_params['noise_distribution'] == 'normal':
        households['aesav'] = x * \
            np.random.normal(loc, scale, size).round(
                2).clip(min=clip_min, max=clip_max)
    else:
        ValueError("Only normal distribution is supported yet.")

    return households


def set_vulnerability(households: pd.DataFrame, is_vulnerability_random: bool, set_vulnerability_params: dict) -> pd.DataFrame:
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
        low = set_vulnerability_params['vulnerability_random_low']  # default 0.01
        high = set_vulnerability_params['vulnerability_random_high']  # default 0.90
        if set_vulnerability_params['vulnerability_random_distribution'] == 'uniform':
            households['v'] = np.random.uniform(
                low, high, households.shape[0])
        else:
            raise ValueError(
                "Only uniform distribution is supported yet.")

    # If vulnerability is not random, use v_init as a starting point and add some noise
    # ?: What is the point of adding the noise to the v_init if we cap it anyhow
    else:
        low = set_vulnerability_params['vulnerability_initial_low']  # default 0.6
        high = set_vulnerability_params['vulnerability_initial_high']  # default 1.4
        # v - actual vulnerability
        # v_init - initial vulnerability
        if set_vulnerability_params['vulnerability_initial_distribution'] == 'uniform':
            households['v'] = households['v_init'] * \
                np.random.uniform(low, high, households.shape[0])
        else:
            raise ValueError(
                "Only uniform distribution is supported yet.")

        # default 0.95
        vulnerability_threshold = set_vulnerability_params['vulnerability_initial_threshold']

        # If vulnerability turned out to be (drawn) is above the threshold, set it to the threshold
        households.loc[households['v']
                       > vulnerability_threshold, 'v'] = vulnerability_threshold

        return households


def calculate_exposure(households: pd.DataFrame, poverty_bias: float, calculate_exposure_params: dict, print_statistics: bool) -> pd.DataFrame:
    '''Calculate exposure of households.

    Exposure is a function of poverty bias, effective capital stock, 
    vulnerability and probable maximum loss.

    Args:
        households (pd.DataFrame): Household survey data for a specific district.
        poverty_bias (float): Poverty bias.
        calculate_exposure_params (dict): Parameters for calculating exposure function.
        print_statistics (bool): If True, print statistics.

    Returns:
        pd.DataFrame: Household survey data with calculated exposure.
    '''
    pml = households['pml'].iloc[0]

    # Random value for poverty bias
    if poverty_bias == 'random':
        low = calculate_exposure_params['poverty_bias_random_low']  # default 0.5
        high = calculate_exposure_params['poverty_bias_random_high']  # default 1.5
        if calculate_exposure_params['poverty_bias_random_distribution'] == 'uniform':
            povbias = np.random.uniform(low, high)
        else:
            raise ValueError(
                "Only uniform distribution is supported yet.")
    else:
        povbias = poverty_bias

    # Set poverty bias to 1 for all households
    households['poverty_bias'] = 1

    # Set poverty bias to povbias for poor households
    households.loc[households['is_poor']
                   == True, 'poverty_bias'] = povbias

    # DEFF: keff - effective capital stock
    delimiter = households[['keff', 'v', 'poverty_bias', 'popwgt']].prod(
        axis=1).sum()

    # ?: fa - fraction affected?
    fa0 = pml / delimiter

    # Print delimiter and fa0 with commas for thousands
    if print_statistics:
        print('PML: ', '{:,}'.format(round(pml, 2)))
        print('Delimiter: ', '{:,}'.format(round(delimiter, 2)))
        print('f0: ', '{:,}'.format(round(fa0, 2)))

    households['fa'] = fa0 * households[['poverty_bias']]
    households.drop('poverty_bias', axis=1, inplace=True)
    return households


def determine_affected(households: pd.DataFrame, determine_affected_params: dict) -> tuple:
    '''Determines affected households.

    We assume that all households have the same probability of being affected, 
    but based on `fa` calculated in `calculate_exposure`.
    '''
    # Get PML, it is the same for all households
    pml = households['pml'].iloc[0]

    # Allow for a relatively small error of 2.5%
    delta = pml * determine_affected_params['delta_pct']  # default 0.025

    # Check if total asset is less than PML
    total_asset = households[['keff', 'popwgt']].prod(axis=1).sum()
    if total_asset < pml:
        raise ValueError(
            'Total asset is less than PML.')

    low = determine_affected_params['low']  # default 0
    high = determine_affected_params['high']  # default 1

    # Generate multiple boolean masks at once
    num_masks = determine_affected_params['num_masks']  # default 1000
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


def apply_individual_policy(households: pd.DataFrame, top_up: int, target_group: str, poverty_line: float) -> pd.DataFrame:
    '''Apply a policy to a specific target group.'''

    # Select a target group
    if target_group == 'all':
        beneficiaries = households['is_affected'] == True

    elif target_group == 'poor':
        beneficiaries = (households['is_affected'] == True) & (
            households['is_poor'] == True)

    elif target_group == 'poor_near_poor1.25':
        beneficiaries = (households['is_affected'] == True) & (
            households['aeexp'] > 1.25 * poverty_line)

    elif target_group == 'poor_near_poor2.0':
        beneficiaries = (households['is_affected'] == True) & (
            households['aeexp'] > 2 * poverty_line)

    # Apply a policy
    households.loc[beneficiaries, 'aesav'] += households.loc[beneficiaries,
                                                             'asset_loss'] * top_up / 100

    # Select columns of interest
    columns_of_interest = ['hhid', 'popwgt', 'own_rent', 'quintile', 'aeexp',
                           'aeexp_house', 'keff', 'v', 'aesav', 'aesoc', 'delta_tax_safety']
    affected_households = households.loc[households['is_affected'],
                                         columns_of_interest].copy()
    return households, affected_households
