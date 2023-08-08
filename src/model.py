# The model function pipeline, which is called by the ema_workbench in `__main__.py`.

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from src.data.read import *
from src.data.write import *
from src.modules.optimize import *
from src.modules.households import *


def load_data(country: str, min_households: int) -> tuple:
    '''Read household survey and asset damage data.

    Args:
        country (str): Country name.
        min_households (int): Minimum number of households that we need to have in a sample to it be representative.

    Returns:
        tuple: Household survey and asset damage files.
    '''

    # Read household survey and asset damage files
    household_survey = read_household_survey(country)
    all_damage = read_asset_damage(country)

    # Duplicate households to have at least `min_households` households
    household_survey = duplicate_households(household_survey, min_households)

    return household_survey, all_damage


def run_model(**kwargs) -> dict:
    '''Run the model.'''
    # TODO: Find the way to document the function input
    # TODO: Find the more concise way to read the model parameters
    # ------------------------- Read the model parameters ------------------------ #
    country = kwargs['country']
    scale = kwargs['scale']
    districts = kwargs['districts']
    min_households = kwargs['min_households']

    # Read household survey and asset damage files
    household_survey, all_damage = load_data(
        country, min_households)

    # Case study constants
    return_period = kwargs['return_period']
    poverty_line = kwargs['poverty_line']
    indigence_line = kwargs['indigence_line']
    saving_rate = kwargs['saving_rate']
    is_vulnerability_random = kwargs['is_vulnerability_random']
    n_years = kwargs['n_years']  # number of years in optimization algorithm

    # Model constants
    estimate_savings_params = kwargs['estimate_savings_params']
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
        my_policy = kwargs['my_policy']
    except:
        # If a policy is not provided, use the default policy
        my_policy = 'all+0'

    add_income_loss = kwargs['add_income_loss']

    # Outcomes
    # Store outcomes in a dictionary, where the key is a district and value is a dictionary of outcomes
    outcomes = {}

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
    # 8. Estimate savings
    # 9. Set vulnerability
    # 10. Calculate exposure
    # 11. Determine affected
    # 12. Apply individual policy
    # 13. Run optimization
    # 14. Integrate wellbeing
    # 15. Prepare outcomes
    # 16. Get outcomes

    # ---------------------- Run the model for each district --------------------- #

    for district in districts:
        # Read household survey and asset damage files for a specific district
        event_damage, total_asset_stock, expected_loss_fraction = get_asset_damage(
            all_damage, scale, district, return_period)

        households = select_district(household_survey, district)

        average_productivity = calculate_average_productivity(households)

        # Model the impact of a disaster on households
        households = (adjust_assets_and_expenditure(households, total_asset_stock, poverty_line, indigence_line)
                      .pipe(calculate_pml, expected_loss_fraction)
                      .pipe(estimate_savings, saving_rate, estimate_savings_params)
                      .pipe(set_vulnerability, is_vulnerability_random, set_vulnerability_params)
                      .pipe(calculate_exposure, poverty_bias, calculate_exposure_params)
                      .pipe(determine_affected, determine_affected_params))

        # households['aesav'].hist()
        # plt.show()

        # Apply a policy
        households, affected_households = apply_individual_policy(
            households, my_policy)

        # Calculate the impact and recovery
        # cash_transfer = {52: 1000, 208: 5000}
        cash_transfer = {}
        affected_households = calculate_recovery_rate(affected_households, consumption_utility, discount_rate, average_productivity, optimization_timestep, n_years)
        affected_households = integrate_wellbeing(affected_households, consumption_utility, discount_rate, income_and_expenditure_growth, average_productivity, poverty_line, n_years, add_income_loss, cash_transfer)
        
        # Add columns of affected households to the original households dataframe
        households = add_columns(households, affected_households)

        # Get outcomes
        array_outcomes = np.array(list(get_outcomes(
            households, event_damage, total_asset_stock, expected_loss_fraction, average_productivity, n_years).values()))

        # * To check whether we have different households affected in different runs
        # if district == 'Castries':
        #     affected_households.to_csv(f'affected_households_{random_seed}.csv')

        outcomes[district] = array_outcomes

    return outcomes