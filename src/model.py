import pandas as pd
import numpy as np
import random
from utils.reader import *
from optimize.optimizer import *
from utils.writer import *
import pickle

def initialize_model(country: str, 
                     scale: str, 
                     min_households: int):
    # Initialize the model variables.

    # * Can do only one country for now
    if country not in ['Saint Lucia']:
        raise ValueError(
            f'Country {country} is not supported. Please use Saint Lucia.')

    # * Can do country only district yet
    if scale not in ['district']:
        raise ValueError(
            f'Scale {scale} is not supported. Please use district.')

    # Call the functions equivalent to the methods _read_household_survey, _duplicate_households, _read_asset_damage
    # Assuming these functions are defined and return necessary variables
    household_survey = read_household_survey(country)
    min_households = min_households
    household_survey = duplicate_households(household_survey, min_households)
    all_damage = read_asset_damage(country)

    # For now, just returning the arguments and parameters as a dictionary
    return household_survey, all_damage

def run_model(**kwargs):
    # Initialize the model variables
    country = kwargs['country']
    scale = kwargs['scale']
    min_households = kwargs['min_households']
    print_statistics = kwargs['print_statistics']
    
    household_survey, all_damage = initialize_model(country, scale, min_households)

    random_seed = kwargs['random_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)

    districts = kwargs['districts']
    return_period = kwargs['return_period']
    poverty_line = kwargs['poverty_line']
    indigence_line = kwargs['indigence_line']

    saving_rate = kwargs['saving_rate']
    assign_savings_params = kwargs['assign_savings_params']

    is_vulnerability_random = kwargs['is_vulnerability_random']
    set_vulnerability_params = kwargs['set_vulnerability_params']
    poverty_bias = kwargs['poverty_bias']
    calculate_exposure_params = kwargs['calculate_exposure_params']
    determine_affected_params = kwargs['determine_affected_params']
    apply_individual_policy_params = kwargs['apply_individual_policy_params']

    my_policy = kwargs['my_policy']

    consumption_utility = kwargs['consumption_utility']
    discount_rate = kwargs['discount_rate']
    optimization_timestep = kwargs['optimization_timestep']
    income_and_expenditure_growth = kwargs['income_and_expenditure_growth']
    x_max = kwargs['x_max']

    outcomes = {}

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
        event_damage, total_asset_stock, expected_loss_fraction = get_asset_damage(all_damage, scale, district, return_period, print_statistics)
        households = select_distict(household_survey, district)
        average_productivity = calculate_average_productivity(households, print_statistics)

        households = (adjust_assets_and_expenditure(households, total_asset_stock, poverty_line, indigence_line, print_statistics)
                        .pipe(calculate_pml, expected_loss_fraction, print_statistics)
                        .pipe(assign_savings, saving_rate, assign_savings_params, print_statistics)
                        .pipe(set_vulnerability, is_vulnerability_random, set_vulnerability_params)
                        .pipe(calculate_exposure, poverty_bias, calculate_exposure_params, print_statistics)
                        .pipe(determine_affected, determine_affected_params, print_statistics))
        households, affected_households = apply_individual_policy(households, my_policy, apply_individual_policy_params)

        average_productivity = calculate_average_productivity(households, print_statistics)
        affected_households = (run_optimization(affected_households, consumption_utility, discount_rate, average_productivity, optimization_timestep)
                                .pipe(integrate_wellbeing, consumption_utility, discount_rate, income_and_expenditure_growth, average_productivity, poverty_line, x_max))
        households = prepare_outcomes(households, affected_households)
        array_outcomes = np.array(list(get_outcomes(households, event_damage, total_asset_stock, expected_loss_fraction, average_productivity, x_max).values()))
        outcomes[district] = array_outcomes

    return outcomes

def select_distict(household_survey:pd.DataFrame, district:str) -> pd.DataFrame:
        '''Select households for a specific district.'''
        return household_survey[household_survey['district'] == district].copy()

def assign_savings(households:pd.DataFrame, saving_rate:float, assign_savings_params:dict, print_statistics:bool) -> pd.DataFrame:
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
    params = assign_savings_params

    # Get the mean of the noise with uniform distribution
    mean_noise_low = params['mean_noise_low']  # default 0
    mean_noise_high = params['mean_noise_high']  # default 5
    if params['mean_noise_distribution'] == 'uniform':
        loc = np.random.uniform(mean_noise_low, mean_noise_high)
    else:
        raise ValueError("Only uniform distribution is supported yet.")

    # Get the scale
    scale = params['noise_scale']  # default 2.5
    size = households.shape[0]
    clip_min = params['savings_clip_min']  # default 0.1
    clip_max = params['savings_clip_max']  # default 1.0

    # Calculate savings with normal noise
    # !: aesav can go to 0 and above 1 because of the mean noise and loc
    # !: See `verifcation.ipynb` for more details
    if params['noise_distribution'] == 'normal':
        households['aesav'] = x * \
            np.random.normal(loc, scale, size).round(
                2).clip(min=clip_min, max=clip_max)
    else:
        ValueError("Only normal distribution is supported yet.")

    if print_statistics:
        print('Minimum expenditure: ', round(
            households['aeexp'].min(), 2))
        print('Maximum expenditure: ', round(
            households['aeexp'].max(), 2))
        print('Average expenditure: ', round(
            households['aeexp'].mean(), 2))
        print('Minimum savings: ', round(
            households['aesav'].min(), 2))
        print('Maximum savings: ', round(
            households['aesav'].max(), 2))
        print('Average savings: ', round(
            households['aesav'].mean(), 2))
    
    return households

def set_vulnerability(households:pd.DataFrame, is_vulnerability_random:bool, set_vulnerability_params:dict) -> pd.DataFrame:
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
    params = set_vulnerability_params

    # If vulnerability is random, then draw from the uniform distribution
    if is_vulnerability_random:
        low = params['vulnerability_random_low']  # default 0.01
        high = params['vulnerability_random_high']  # default 0.90
        if params['vulnerability_random_distribution'] == 'uniform':
            households['v'] = np.random.uniform(
                low, high, households.shape[0])
        else:
            raise ValueError(
                "Only uniform distribution is supported yet.")

    # If vulnerability is not random, use v_init as a starting point and add some noise
    # ?: What is the point of adding the noise to the v_init if we cap it anyhow
    else:
        low = params['vulnerability_initial_low']  # default 0.6
        high = params['vulnerability_initial_high']  # default 1.4
        # v - actual vulnerability
        # v_init - initial vulnerability
        if params['vulnerability_initial_distribution'] == 'uniform':
            households['v'] = households['v_init'] * \
                np.random.uniform(low, high, households.shape[0])
        else:
            raise ValueError(
                "Only uniform distribution is supported yet.")

        # default 0.95
        vulnerability_threshold = params['vulnerability_initial_threshold']

        # If vulnerability turned out to be (drawn) is above the threshold, set it to the threshold
        households.loc[households['v']
                            > vulnerability_threshold, 'v'] = vulnerability_threshold
        
        return households

def calculate_exposure(households: pd.DataFrame, poverty_bias: float, calculate_exposure_params: dict, print_statistics) -> pd.DataFrame:
    '''Calculate exposure of households.

    Exposure is a function of poverty bias, effective captial stock, 
    vulnerability and probable maximum loss.
    '''
    pml = households['pml'].iloc[0]
    params = calculate_exposure_params

    # Random value for poverty bias
    if poverty_bias == 'random':
        low = params['poverty_bias_random_low']  # default 0.5
        high = params['poverty_bias_random_high']  # default 1.5
        if params['poverty_bias_random_distribution'] == 'uniform':
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

    households['fa'] = fa0*households[['poverty_bias']]

    # !: households['fa'] seems to be the same for all households
    households.drop('poverty_bias', axis=1, inplace=True)
    return households

def determine_affected(households: pd.DataFrame, determine_affected_params: dict, print_statistics: bool) -> pd.DataFrame:
    '''Determine affected households.


    '''
    params = determine_affected_params
    low = params['low']  # default 0
    high = params['high']  # default 1

    if params['distribution'] == 'uniform':
        # !: This is very random
        households['is_affected'] = households['fa'] >= np.random.uniform(
            low, high, households.shape[0])
    else:
        raise ValueError("Only uniform distribution is supported yet.")

    affected_households = households[households['is_affected'] == True]
    total_asset = households[['keff', 'popwgt']].prod(axis=1).sum()
    total_asset_loss = affected_households[['keff', 'v', 'popwgt']].prod(axis=1).sum()

    if print_statistics:
        n_affected = households['is_affected'].multiply(
            households['popwgt']).sum()
        fraction_affected = n_affected / households['popwgt'].sum()
        print('Total number of households: ', '{:,}'.format(
            round(households['popwgt'].sum())))
        print('Number of affected households: ',
                '{:,}'.format(round(n_affected)))
        print(
            f'Fraction of affected households: {round((fraction_affected * 100), 2)}%')
        print('Total asset: ', '{:,}'.format(round(total_asset)))
        print('Total asset loss: ', '{:,}'.format(round(total_asset_loss)))

    # TODO: Create model construction with bifurcate option
    return households

def apply_individual_policy(households:pd.DataFrame, my_policy: str, apply_individual_policy_params: dict) -> pd.DataFrame:
        households['DRM_cost'] = 0
        households['DRM_cash'] = 0

        if my_policy == 'None':
            households['DRM_cost'] = 0
            households['DRM_cash'] = 0

        elif my_policy == 'Existing_SP_100':
            # Beneficiaries are affected households
            beneficiaries = households['is_affected'] == True

            # Assign `DRM_cost`` to `aesoc` to beneficiaries, for the rest 0
            households.loc[beneficiaries,
                                'DRM_cost'] = households.loc[beneficiaries, 'aesoc']
            households['DRM_cost'] = households['DRM_cost'].fillna(0)

            # Assign `DRM_cash` to `aesoc` for beneficiaries, for the rest 0
            households.loc[beneficiaries,
                                'DRM_cash'] = households.loc[beneficiaries, 'aesoc']
            households['DRM_cash'] = households['DRM_cash'].fillna(0)

            # Increase `aesav` by `aesoc`
            households.loc[beneficiaries,
                                'aesav'] += households.loc[beneficiaries, 'aesoc']

        elif my_policy == 'Existing_SP_50':
            # Beneficiaries are those who are affected
            beneficiaries = households['is_affected'] == True

            # Assign `DRM_cost`` to 0.5 * `aesoc` to beneficiaries, for the rest 0
            households.loc[beneficiaries,
                                'DRM_cost'] = households.loc[beneficiaries, 'aesoc'] * 0.5
            households['DRM_cost'] = households['DRM_cost'].fillna(0)

            # Assign `DRM_cash` to 0.5 * `aesoc` to beneficiaries, for the rest 0
            households.loc[beneficiaries,
                                'DRM_cash'] = households.loc[beneficiaries, 'aesoc'] * 0.5
            households['DRM_cash'] = households['DRM_cash'].fillna(0)

            # Increase `aesav` by 0.5 `aesoc`
            households.loc[beneficiaries,
                                'aesav'] += households.loc[beneficiaries, 'aesoc'] * 0.5

        elif my_policy == 'retrofit':
            params = apply_individual_policy_params
            a = params['retrofit_a']  # default 0.05
            b = params['retrofit_b']  # default 0.7
            c = params['retrofit_c']  # default 0.2
            clip_lower = params['retrofit_clip_lower']  # default 0
            clip_upper = params['retrofit_clip_upper']  # default 0.7
            households['DRM_cost'] = a * households[['keff', 'aewgt']
                                                              ].prod(axis=1) * ((households['v'] - b) / c).clip(lower=clip_lower)
            households['DRM_cash'] = 0
            households['v'] = households['v'].clip(upper=clip_upper)

        elif my_policy == 'retrofit_roof1':
            params = apply_individual_policy_params
            # default [2, 4, 5, 6]
            roof_material_of_interest = params['retrofit_roof1_roof_materials_of_interest']
            # Beneficiaries are those who have roof of a certain material
            beneficiaries = households['roof_material'].isin(
                roof_material_of_interest)

            a = params['retrofit_roof1_a']  # default 0.05
            b = params['retrofit_roof1_b']  # default 0.1
            c = params['retrofit_roof1_c']  # default 0.2
            d = params['retrofit_roof1_d']  # default 0.1

            households.loc[beneficiaries, 'DRM_cost'] = a * \
                households['keff'] * (b / c)

            households.loc[beneficiaries, 'DRM_cash'] = 0

            # Decrease vulnerability `v` by `d`
            households.loc[beneficiaries, 'v'] -= d

        elif my_policy == 'PDS':
            # Benefiaries are those who are affected and have their own house
            beneficiaries = (households['is_affected'] == True) & (
                households['own_rent'] == 'own')
            # accounting
            households.loc[beneficiaries, 'DRM_cost'] = households.loc[beneficiaries].eval(
                'keff*v')
            households['DRM_cost'] = households['DRM_cost'].fillna(
                0)
            households.loc[beneficiaries, 'DRM_cash'] = households.loc[beneficiaries].eval(
                'keff*v')
            households['DRM_cash'] = households['DRM_cash'].fillna(
                0)

            # Increase `aesav` by `keff*v`
            households.loc[beneficiaries,
                                'aesav'] += households.loc[beneficiaries].eval('keff*v')

        else:
            raise ValueError(
                'Policy not found. Please use one of the following: None, Existing_SP_100, Existing_SP_50, retrofit, retrofit_roof1 or PDS.')

        columns_of_interest = ['hhid',
                               'popwgt',
                               'own_rent',
                               'quintile',
                               'aeexp',
                               'aeexp_house',
                               'keff',
                               'v',
                               'aesav',
                               'aesoc',
                               'delta_tax_safety']

        affected_households = households.loc[households['is_affected'], columns_of_interest].copy(
        )
        return households, affected_households