# The model function pipeline, which is called by the ema_workbench in `__main__.py`.

import pandas as pd
import numpy as np
import random
from unbreakable.data.reader import *
from unbreakable.data.writer import *
from unbreakable.modules.optimizer import *
from unbreakable.modules.households import *


def run_model(**params) -> dict:
    '''Run the simulation model.

    Args:
        params (dict): A dictionary of parameters. For a complete list of parameters, see `config/SaintLucia.yaml`.

    Returns:
        dict: A dictionary of outcomes. The key is a district and value is a dictionary of outcomes.
    '''
    # ------------------------------ Read parameters ----------------------------- #
    # Case study parameters
    country = params['country']
    districts = params['districts']

    # TODO: Do something with duplicate_households function
    min_households = params['min_households']

    # Read household survey and asset damage files

    all_households = read_household_survey(country)
    
    # TODO: Check what do we actually read here
    all_damage = read_asset_damage(country)

    # Case study constants
    return_period = params['return_period']
    poverty_line = params['poverty_line']
    indigence_line = params['indigence_line']
    saving_rate = params['saving_rate']
    is_vulnerability_random = params['is_vulnerability_random']
    n_years = params['n_years']

    # Model constants
    estimate_savings_params = params['estimate_savings_params']
    assign_vulnerability_params = params['assign_vulnerability_params']
    calculate_exposure_params = params['calculate_exposure_params']
    identify_affected_params = params['identify_affected_params']

    # Uncertainties
    poverty_bias = params['poverty_bias']
    consumption_utility = params['consumption_utility']
    discount_rate = params['discount_rate']
    optimization_timestep = params['optimization_timestep']
    income_and_expenditure_growth = params['income_and_expenditure_growth']

    # Policy levers
    try:
        my_policy = params['my_policy']
    except:
        # If a policy is not provided, use the default policy
        my_policy = 'all+0'

    # Add income loss to consumption loss calculation
    add_income_loss = params['add_income_loss']

    # Fix random seed for reproducibility
    random_seed = params['random_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Outcomes
    # Store outcomes in a dictionary, where the key is a district and value is a dictionary of outcomes
    outcomes = {}

    # ---------------------- Run the model for each district --------------------- #

    for district in districts:
        # Get total exposed asset stock and expected loss fraction for a specific district and return period
        total_exposed_asset_stock = get_total_exposed_asset_stock(all_damage, district, return_period)
        expected_loss_fraction = get_expected_loss_fraction(all_damage, district, return_period)

        # Select households in a specific district
        households = all_households[all_households['district'] == district].copy(
        )

        # Model the impact of a disaster on households
        # TODO: Rename `n_years` to something more meaningful

        # Calculate the impact and recovery
        # cash_transfer = {52: 1000, 208: 5000}
        cash_transfer = {}

        households = (households.pipe(calculate_median_productivity)
                                .pipe(adjust_assets_and_expenditure, total_exposed_asset_stock, poverty_line, indigence_line)
                                .pipe(calculate_household_probable_maximum_loss, expected_loss_fraction)
                                .pipe(estimate_savings, saving_rate, estimate_savings_params)
                                .pipe(assign_vulnerability, is_vulnerability_random, assign_vulnerability_params)
                                .pipe(calculate_exposure, poverty_bias, calculate_exposure_params)
                                .pipe(identify_affected, identify_affected_params)
                                .pipe(apply_policy, my_policy)
                                .pipe(calculate_recovery_rate, consumption_utility, discount_rate, optimization_timestep, n_years)
                                .pipe(integrate_wellbeing, consumption_utility, discount_rate, income_and_expenditure_growth, poverty_line, n_years, add_income_loss, cash_transfer))

        # Get outcomes
        array_outcomes = np.array(list(get_outcomes(
            households, total_exposed_asset_stock, expected_loss_fraction, n_years).values()))

        # * To check whether we have different households affected in different runs
        # if district == 'Castries':
        #     affected_households.to_csv(f'affected_households_{random_seed}.csv')

        outcomes[district] = array_outcomes

    return outcomes
