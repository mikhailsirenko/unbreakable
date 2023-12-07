"""This module contains the simulation model.
It initializes it, loads the data and defines the pipeline. 
The pipeline is then called/ran by the EMA Workbench in `run.py`"""

import numpy as np
import random
from pathlib import Path
import pandas as pd
from unbreakable.data.reader import *
from unbreakable.data.writer import *
from unbreakable.modules.shock import *
from unbreakable.modules.households import *
from unbreakable.modules.policy import *
from unbreakable.modules.recovery import *
from ema_workbench import *


def model(**params) -> dict:
    '''Simulation model.

    Args:
        params (dict): A dictionary of parameters. For a complete list of parameters, see `config/SaintLucia.yaml`.

    Returns:
        dict: A dictionary of outcomes. The key is a district and value is a dictionary of outcomes.
    '''
    # Validate parameters

    # Case study parameters
    country = params['country']
    districts = params['districts']

    # Read household survey and damage files
    all_households = read_household_survey(country)
    all_damage = read_asset_damage(country)

    # Case study constants
    return_period = params['return_period']
    years_to_recover = params['years_to_recover']
    average_productivity = params['average_productivity']

    # Model constants
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

    w = calculate_w(all_households, consump_util)

    for district in districts:
        # Get expected loss fraction total exposed asset stock for a specific district and return period
        loss_fraction, exposed_stock = get_loss_fraction_and_exposed_stock(
            all_damage, district, return_period)

        # Select households in a specific district
        households = all_households[all_households['district'] == district].copy(
        )

        # For the dynamic policy
        # cash_transfer = {52: 1000, 208: 5000}
        cash_transfer = {}

        # Store average productivity in households for the sake of simplicity
        households['average_productivity'] = average_productivity

        households = (households.pipe(match_assets_and_damage, exposed_stock, atol)
                                .pipe(calculate_pml, loss_fraction)
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
            households, exposed_stock, loss_fraction, years_to_recover, w).values()))

        # * To check whether we have different households affected in different runs
        # if district == 'Castries':
        #     affected_households.to_csv(f'affected_households_{random_seed}.csv')

        outcomes[district] = array_outcomes

    return outcomes


def load_config(country: str) -> dict:
    """
    Load configuration for the specified country.

    Args:
        country (str): The name of the country for which to load the configuration.

    Returns:
        dict: Configuration dictionary loaded from the YAML file.

    Raises:
        FileNotFoundError: If the YAML configuration file for the specified country is not found.
    """
    config_path = Path(f"../config/{country}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file for {country} not found at {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def setup_model(config: dict) -> Model:
    """
    Set up the EMA Workbench model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary loaded from the YAML file.

    Returns:
        Model: Configured EMA Workbench model.
    """
    # Initialize the EMA Workbench model
    my_model = Model(name="model", function=model)

    # Extract and set up uncertainties, constants, and levers from the config
    uncertainties = config.get("uncertainties", {})
    constants = config.get("constants", {})
    levers = config.get("levers", {})

    # Define seed as an uncertainty for multiple runs
    seed_start = 0
    seed_end = 1000000000
    my_model.uncertainties = [IntegerParameter("random_seed", seed_start, seed_end)]\
        #   + [RealParameter(key, values[0], values[1]) for key, values in uncertainties.items()]

    # Constants
    my_model.constants = [Constant(key, value)
                          for key, value in constants.items()]

    # Levers
    my_model.levers = [CategoricalParameter(
        key, values) for key, values in levers.items()]

    # Outcomes
    my_model.outcomes = [ArrayOutcome(district)
                         for district in constants.get('districts', [])]

    return my_model


def run_experiments(model: Model, n_scenarios: int, n_policies: int, country: str, n_processes: int = 12):
    """
    Run experiments on the model using EMA Workbench.

    Args:
        model (Model): The EMA Workbench model to run.
        n_scenarios (int): Number of scenarios to run.
        n_policies (int): Number of policies to run.
        country (str): Name of the country for which the experiments are run.
        n_processes (int): Number of processes to use for multiprocessing. Defaults to 12.
    """
    with MultiprocessingEvaluator(model, n_processes=n_processes) as evaluator:
        results = evaluator.perform_experiments(
            scenarios=n_scenarios, policies=n_policies)

    results_path = Path(f'../experiments/{country}')
    results_path.mkdir(parents=True, exist_ok=True)
    save_results(results, results_path /
                 f"scenarios={n_scenarios}, policies={n_policies}.tar.gz")


def validate_params(params: dict) -> None:
    '''
    Validates that all required parameters are present.

    Args:
        params (dict): Dictionary containing parameters.

    Raises:
        ValueError: If any required parameter is missing.
    '''

    required_keys = [
        'country', 'districts', 'return_period', 'years_to_recover', 'average_productivity',
        'consump_util', 'discount_rate', 'lambda_increment', 'income_and_expenditure_growth',
        'add_income_loss', 'random_seed', 'atol', 'poverty_bias', 'calc_exposure_params',
        'ident_affected_params', 'save_households', 'my_policy'
    ]

    missing_keys = [key for key in required_keys if key not in params]

    if missing_keys:
        raise ValueError(
            f"Missing essential parameters: {', '.join(missing_keys)}")


def calculate_w(all_households: pd.DataFrame, consump_util: float) -> float:
    '''
    Calculates w factor.
    '''
    return (np.sum(all_households['aeexp'] * all_households['wgt']) / np.sum(all_households['wgt']))**(-consump_util)


def save_households_data(households: pd.DataFrame, country: str, district: str, random_seed: int):
    '''
    Saves the households data to a CSV file.
    '''
    output_dir = Path(f'../experiments/{country}/households/')
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f'{district}_{random_seed}.csv'
    households.to_csv(file_path)
