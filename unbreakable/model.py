"""This module contains the simulation model.
It initializes it, loads the data and defines the pipeline. 
The pipeline is then called/ran by the EMA Workbench in `run.py`"""

import numpy as np
import random
from pathlib import Path
import pandas as pd
from unbreakable.data.reader import *
from unbreakable.data.writer import *
from unbreakable.modules.integrator import *
from unbreakable.modules.households import *


def model(**params) -> dict:
    '''Simulation model.

    Args:
        params (dict): A dictionary of parameters. For a complete list of parameters, see `config/SaintLucia.yaml`.

    Returns:
        dict: A dictionary of outcomes. The key is a district and value is a dictionary of outcomes.
    '''
    # Case study parameters
    country = params['country']
    districts = params['districts']
    min_representative_households = params['min_representative_households']

    # Read household survey and damage files
    all_households = read_household_survey(country)
    all_damage = read_asset_damage(country)

    # Case study constants
    return_period = params['return_period']
    poverty_line = params['poverty_line']
    indigence_line = params['indigence_line']
    saving_rate = params['saving_rate']
    is_vuln_random = params['is_vuln_random']
    years_to_recover = params['years_to_recover']
    average_productivity = params['average_productivity']

    # Model constants
    est_sav_params = params['est_sav_params']
    assign_vuln_params = params['assign_vuln_params']
    calc_exposure_params = params['calc_exposure_params']
    ident_affected_params = params['ident_affected_params']
    save_households = params['save_households']
    # Abs tolerance for matching asset stock of damage and household survey data sets
    atol = params['atol']

    # Uncertainties
    poverty_bias = params['poverty_bias']
    consump_util = params['consump_util']
    discount_rate = params['discount_rate']
    lambda_increment = params['lambda_increment']
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

    # Calculate the wprime for the whole country
    wprime = calculate_wprime(all_households, all_damage, districts, return_period,
                              min_representative_households, random_seed, poverty_line, indigence_line, atol, consump_util)

    for district in districts:
        # Get total exposed asset stock and expected loss fraction for a specific district and return period
        tot_exposed_asset = get_tot_exposed_asset_stock(
            all_damage, district, return_period)
        expected_loss_frac = get_expected_loss_frac(
            all_damage, district, return_period)

        # Select households in a specific district
        households = all_households[all_households['district'] == district].copy(
        )

        # For the dynamic policy
        # cash_transfer = {52: 1000, 208: 5000}
        cash_transfer = {}

        households['average_productivity'] = average_productivity

        households = (households.pipe(duplicate_households, min_representative_households, random_seed)
                                .pipe(match_assets_and_expenditure, tot_exposed_asset, poverty_line, indigence_line, atol)
                                .pipe(calculate_district_pml, expected_loss_frac)
                                .pipe(estimate_savings, saving_rate, est_sav_params)
                                .pipe(assign_vulnerability, is_vuln_random, assign_vuln_params)
                                .pipe(calculate_exposure, poverty_bias, calc_exposure_params)
                                .pipe(identify_affected, ident_affected_params)
                                .pipe(apply_policy, my_policy)
                                .pipe(calculate_recovery_rate, consump_util, discount_rate, lambda_increment, years_to_recover)
                                .pipe(calculate_wellbeing, consump_util, discount_rate, income_and_expenditure_growth, years_to_recover, add_income_loss, cash_transfer))

        if save_households:
            Path(f'../experiments/households/').mkdir(parents=True, exist_ok=True)
            households.to_csv(
                f'../experiments/households/{district}_{random_seed}.csv')

        array_outcomes = np.array(list(get_outcomes(
            households, tot_exposed_asset, expected_loss_frac, years_to_recover, wprime).values()))

        # * To check whether we have different households affected in different runs
        # if district == 'Castries':
        #     affected_households.to_csv(f'affected_households_{random_seed}.csv')

        outcomes[district] = array_outcomes

    return outcomes


def initialize_model(country: str, scale: str):
    # First check whether all necessary folders exist
    Path(f'../experiments/').mkdir(parents=True, exist_ok=True)
    Path(f'../experiments/households/').mkdir(parents=True, exist_ok=True)

    # Check whether the data folder exists
    if not Path(f'../data/processed/').exists():
        raise Exception('The "../data/processed/" folder does not exist.')

    else:
        if not Path(f'../data/processed/asset_damage/{country}.xlsx').exists():
            raise Exception(
                f'The "../data/processed/asset_damage/{country}.xlsx" file does not exist.')

        if not Path(f'../data/processed/household_survey/{country}.xlsx').exists():
            raise Exception(
                f'The "../data/processed/household_survey/{country}.xlsx" file does not exist.')

    # Check whether data files have necessary columns
    columns = ['rp', 'district',
               'total_exposed_asset_stock', 'probable_maximum_loss']
    df = pd.read_excel(f'../data/processed/asset_damage/{country}.xlsx')
    if not all([column in df.columns for column in columns]):
        raise Exception(
            f'The "../data/processed/asset_damage/{country}.xlsx" file does not have all necessary columns.')

    columns = []

    # Check whether config file exists


def check_folders():
    pass


def check_files():
    pass


def check_columns():
    pass
