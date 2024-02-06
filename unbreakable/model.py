import numpy as np
import random
from pathlib import Path
import pandas as pd
import yaml
from pathlib import Path
from ema_workbench import *
from unbreakable.data.reader import *
from unbreakable.analysis.calculator import *
from unbreakable.modules.disaster import *
from unbreakable.modules.households import *
from unbreakable.modules.policy import *
from unbreakable.modules.recovery import *


def model(**params) -> dict:
    '''Simulation model.

    Args:
        params (dict): A dictionary of parameters. For a complete list of parameters, see `config/SaintLucia.yaml`.

    Returns:
        dict: A dictionary of outcomes. The key is a district and value is a dictionary of outcomes.
    '''
    # Validate parameters
    validate_params(params)

    # If a policy is not provided, use the default policy
    my_policy = params.get('my_policy', 'all+0')

    # Fix random seed
    random.seed(params['random_seed'])
    np.random.seed(params['random_seed'])

    # Read household survey and damage files
    all_households = read_household_survey(params['country'])
    all_risk_and_damage = read_disaster_risk_data(params['country'])

    # Store average productivity in the household dataframe for the sake of simplicity
    all_households['average_productivity'] = params['average_productivity']

    # Store outcomes in a dictionary, where the key is a district and value is a dictionary of outcomes
    outcomes = {}

    # Calculate w factor
    w = (np.sum(all_households['aeexp'] * all_households['wgt']
                ) / np.sum(all_households['wgt']))**(-params['consump_util'])

    for district in params['districts']:
        # Select households in a specific district
        district_households = all_households[all_households['district'] == district].copy(
        )

        # Get disaster data for the district and return period
        exposed_stock, loss_fraction, pml = get_stock_and_loss_fraction(
            all_risk_and_damage, district, params['return_period'])

        # Store pml in the household dataframe for the sake of simplicity
        district_households['pml'] = pml

        # For the dynamic policy
        # cash_transfer = {52: 1000, 208: 5000}
        cash_transfer = {}

        district_households = (district_households.pipe(calculate_exposure, params['poverty_bias'], params['calc_exposure_params'])
                                                  .pipe(identify_affected, params['ident_affected_params'])
                                                  .pipe(apply_policy, my_policy)
                                                  .pipe(calculate_recovery_rate, params['consump_util'], params['discount_rate'], params['lambda_increment'], params['years_to_recover'])
                                                  .pipe(calculate_wellbeing, params['consump_util'], params['discount_rate'], params['income_and_expenditure_growth'], params['years_to_recover'], params['add_income_loss'], cash_transfer))

        if params['save_households']:
            save_households(
                district_households, params['country'], district, params['random_seed'])

        outcomes[district] = np.array(list(calculate_outcomes(
            district_households, exposed_stock, loss_fraction, params['years_to_recover'], w).values()))

    return outcomes


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
        'ident_affected_params', 'save_households'
    ]

    missing_keys = [key for key in required_keys if key not in params]

    if missing_keys:
        raise ValueError(
            f"Missing essential parameters: {', '.join(missing_keys)}")


def save_households(households: pd.DataFrame, country: str, district: str, random_seed: int):
    '''
    Saves the households data to a CSV file.
    '''
    output_dir = Path(f'../experiments/{country}/households/')
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f'{district}_{random_seed}.csv'
    households.to_csv(file_path)


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


def run_experiments(model: Model, n_scenarios: int, n_policies: int, country: str, multiprocessing: bool, n_processes: int = 12):
    """
    Run experiments on the model using EMA Workbench and save results.

    Args:
        model (Model): The EMA Workbench model to run.
        n_scenarios (int): Number of scenarios to run.
        n_policies (int): Number of policies to run.
        country (str): Name of the country for which the experiments are run.
        multiprocessing (bool): Whether to use multiprocessing.
        n_processes (int): Number of processes to use for multiprocessing. Defaults to 12.
    """
    if multiprocessing:
        with MultiprocessingEvaluator(model, n_processes=n_processes) as evaluator:
            results = evaluator.perform_experiments(
                scenarios=n_scenarios, policies=n_policies)
    else:
        results = perform_experiments(
            models=model, scenarios=n_scenarios, policies=n_policies)

    results_path = Path(f'../experiments/{country}')
    results_path.mkdir(parents=True, exist_ok=True)
    save_results(results, results_path /
                 f"scenarios={n_scenarios}, policies={n_policies}.tar.gz")
