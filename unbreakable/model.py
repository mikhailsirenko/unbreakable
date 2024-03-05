import numpy as np
from pathlib import Path
import pandas as pd
import yaml
from pathlib import Path
from ema_workbench import *
from unbreakable.data.reader import *
from unbreakable.data.randomizer import *
from unbreakable.analysis.calculator import *
from unbreakable.modules.disaster import *
from unbreakable.modules.households import *
from unbreakable.modules.policy import *
from unbreakable.modules.recovery import *
from unbreakable.modules.conflict import *


def model(**params) -> dict:
    '''Simulation model.

    Args:
        params (dict): A dictionary of parameters. For a complete list of parameters, see `config/SaintLucia.yaml`.

    Returns:
        dict: A dictionary of outcomes. The key is a district and value is a dictionary of outcomes.
    '''
    # Make sure that we have all required global parameters
    validate_params(params)

    # If a policy is not provided, use the default policy
    current_policy = params.get('current_policy', 'none')
    country = params['country']

    # Fix random seed for reproducibility
    random_seed = params['random_seed']
    np.random.seed(random_seed)

    # Read household survey and damage files
    all_households = read_household_survey(params['country'])
    risk_and_damage = read_risk_and_damage(params['country'])

    # Randomize households survey data
    all_households = randomize(
        all_households, risk_and_damage, params, random_seed=random_seed)

    # Save random seed for reproducibility
    all_households['random_seed'] = random_seed

    # Store outcomes in a dictionary, where the key is a district and value is a dictionary of outcomes
    outcomes = {}

    # Whether country has a conflict
    is_conflict = params['is_conflict']

    welfare = calculate_welfare(all_households, params['cons_util'])

    # If there is a conflict, read conflict data and calculate its effect on the economy
    if is_conflict:
        conflict_impact = read_conflict_data(params['country'])
        affected_economy = affect_economy(
            conflict_impact, params['avg_prod'], params['inc_exp_growth'])

    # If there is no conflict, use the base values specified in the config file
    else:
        avg_prod = params['avg_prod']
        inc_exp_growth = params['inc_exp_growth']

    for region in params['regions']:
        # Select households in a specific region
        region_households = all_households[all_households['region'] == region].copy(
        )

        # Get disaster data for the region and return period
        exposed_stock, loss_fraction, region_pml = get_region_damage(
            risk_and_damage, region, params['return_per'])

        # Get affected average productivity and income and expenditure growth for the region
        if is_conflict:
            avg_prod = affected_economy[affected_economy['region']
                                        == region]['avg_prod'].values[0]
            inc_exp_growth = affected_economy[affected_economy['region']
                                              == region]['inc_exp_growth'].values[0]

        # For the dynamic policy
        # cash_transfer = {52: 1000, 208: 5000}
        cash_transfer = {}

        region_households = (region_households
                             .pipe(estimate_impact, region_pml, params['pov_bias'], params['calc_exposure_params'])
                             .pipe(identify_affected, region_pml, params['identify_aff_params'], random_seed=random_seed)
                             .pipe(apply_policy, country, current_policy, random_seed)
                             .pipe(calc_rec_rate, avg_prod, params['cons_util'], params['disc_rate'], params['lambda_incr'], params['yrs_to_rec'])
                             .pipe(calc_wellbeing, avg_prod, params['cons_util'], params['disc_rate'], inc_exp_growth, params['yrs_to_rec'], params['add_inc_loss'], cash_transfer, is_conflict))

        if params['save_households']:
            save_households(
                region_households, params['country'], region, random_seed)

        outcomes[region] = np.array(list(calculate_outcomes(
            region_households, exposed_stock, loss_fraction, region_pml, params['yrs_to_rec'], welfare).values()))

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
        'country', 'regions', 'return_per', 'yrs_to_rec', 'avg_prod',
        'cons_util', 'disc_rate', 'lambda_incr', 'inc_exp_growth',
        'add_inc_loss', 'random_seed', 'atol', 'pov_bias', 'calc_exposure_params',
        'identify_aff_params', 'is_conflict', 'save_households'
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


def setup_model(config: dict, n_replications: int = None) -> Model:
    """
    Set up the EMA Workbench model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary loaded from the YAML file.
        n_replications (int): Number of replications for the model. Defaults to None.

    Returns:
        Model: Configured EMA Workbench model.
    """
    # Initialize the EMA Workbench model
    my_model = Model(name="model", function=model)
    # my_model = ReplicatorModel(name="model", function=model)

    # Extract and set up uncertainties, constants, and levers from the config
    # uncertainties = config.get("uncertainties", {})
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
        'current_policy', [values for _, values in levers.items()])]

    # Outcomes
    my_model.outcomes = [ArrayOutcome(region)
                         for region in constants.get('regions', [])]

    # my_model.replications = n_replications

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

    try:
        is_conflict = model.constants._data['is_conflict'].value
    except KeyError:
        is_conflict = False

    if is_conflict:
        save_results(results, results_path /
                     f"scenarios={n_scenarios}, policies={n_policies}, conflict=True.tar.gz")
    else:
        save_results(results, results_path /
                     f"scenarios={n_scenarios}, policies={n_policies}.tar.gz")
