import numpy as np
from pathlib import Path
import pandas as pd
import yaml
from pathlib import Path
from typing import NamedTuple
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
    replicator = params.get('replicator', False)
    if not replicator:
        random_seed = params['random_seed']
        np.random.seed(random_seed)
    else:
        random_seed = None

    check_params(params)
    households, risk_and_damage = load_data(params['country'])
    households = randomize(
        households, risk_and_damage, params, random_seed=random_seed)

    if params['is_conflict']:
        conflict = read_conflict_data(params['country'])
    else:
        conflict = None

    welfare = calculate_welfare(households, params['cons_util'])

    outcomes = {}

    for region in params['regions']:
        region_households = households[households['region'] == region].copy(
        )

        total_exposed_stock, expected_loss_fraction, region_pml = get_region_damage(
            risk_and_damage, region, params['return_period'])

        if params['is_conflict']:
            conflict_intensity = conflict[conflict['region']
                                          == region]['Bin'].values[0]
        else:
            conflict_intensity = None

        region_households = (region_households
                             .pipe(calculate_exposure, region_pml, params['pov_bias'], params['calc_exposure_params'])
                             .pipe(identify_affected, region_pml, params['identify_aff_params'], random_seed=random_seed)
                             .pipe(apply_policy, params['country'], params.get('current_policy', 'none'), params['disaster_type'], random_seed=random_seed)
                             .pipe(calculate_recovery_rate, params['avg_prod'], params['cons_util'], params['disc_rate'], params['lambda_incr'], params['yrs_to_rec'])
                             .pipe(calculate_wellbeing, params['avg_prod'], params['cons_util'], params['disc_rate'], params['inc_exp_growth'], params['yrs_to_rec'], params['add_inc_loss'], params['save_consumption_recovery'], params['is_conflict'], conflict_intensity))

        if params['save_households']:
            save_households(
                params['country'], region, region_households, random_seed=random_seed)

        outcomes[region] = np.array(list(calculate_outcomes(
            region_households, total_exposed_stock, expected_loss_fraction, region_pml, params['yrs_to_rec'], welfare).values()))

    return outcomes


def check_params(params: dict) -> None:
    '''Check that all config parameters for a case study are present.'''
    required_params = ['country', 'regions', 'avg_prod',
                       'cons_util', 'disc_rate',
                       'calc_exposure_params', 'identify_aff_params',
                       'add_inc_loss', 'pov_bias', 'lambda_incr', 'yrs_to_rec',
                       'rnd_house_vuln_params', 'rnd_inc_params', 'rnd_sav_params',
                       'rnd_rent_params', 'min_households', 'atol', 'save_households']
    missing_params = [
        param for param in required_params if param not in params]
    if missing_params:
        raise ValueError(
            f"Missing essential parameters: {', '.join(missing_params)}")


def save_households(households: pd.DataFrame, params: dict, random_seed: int):
    '''Save region households data to a CSV file.'''
    country = params['country']
    region = params['region']
    output_dir = Path(f'../experiments/{country}/households/')
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f'{region}_{random_seed}.csv'
    households.to_csv(file_path)


def load_config(country: str, return_period: int) -> dict:
    '''Load configuration for the specified case country.'''
    config_path = Path(f"../config/{country}.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file for {country} not found at {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return_periods = [10, 50, 100, 250, 500, 1000]
    if return_period not in return_periods:
        raise ValueError(
            f"Return period {return_period} not in available return periods: {return_periods}")
    config['constants']['return_period'] = return_period
    return config


def setup_model(config: dict, replicator: bool) -> Model:
    """
    Set up the EMA Workbench model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary loaded from the YAML file.

    Returns:
        Model: Configured EMA Workbench model.
    """
    # Initialize the EMA Workbench model
    if replicator:
        # Set up the ReplicatorModel to iterate over multiple seeds
        my_model = ReplicatorModel(name="model", function=model)

        # Extract and set up uncertainties, constants, and levers from the config
        # uncertainties = config.get("uncertainties", {})
        constants = config.get("constants", {})
        levers = config.get("levers", {})

        # Uncertainties
        my_model.uncertainties = [
            CategoricalParameter('is_conflict', [False, True])]

        # Constants
        my_model.constants = [Constant(key, value)
                              for key, value in constants.items()] + [Constant('replicator', True)]

        # Levers
        my_model.levers = [CategoricalParameter(
            'current_policy', [values for _, values in levers.items()])]

        # Outcomes
        my_model.outcomes = [ArrayOutcome(region)
                             for region in constants.get('regions', [])]

        my_model.replications = 2

        return my_model
    else:
        my_model = Model(name="model", function=model)

        # Extract and set up uncertainties, constants, and levers from the config
        # uncertainties = config.get("uncertainties", {})
        constants = config.get("constants", {})
        levers = config.get("levers", {})

        # Define seed as an uncertainty for multiple runs
        seed_start = 0
        seed_end = 1000000000

        my_model.uncertainties = [IntegerParameter(
            "random_seed", seed_start, seed_end)]

        # Constants
        my_model.constants = [Constant(key, value)
                              for key, value in constants.items()]

        # Levers
        my_model.levers = [CategoricalParameter(
            'current_policy', [values for _, values in levers.items()])]

        # Outcomes
        my_model.outcomes = [ArrayOutcome(region)
                             for region in constants.get('regions', [])]

        return my_model


def run_experiments(experimental_setup: dict) -> None:
    """Run experiments on the model using EMA Workbench and save results."""
    country = experimental_setup['country']
    return_period = experimental_setup['return_period']
    model = experimental_setup['model']
    n_scenarios = experimental_setup['n_scenarios']
    n_policies = experimental_setup['n_policies']
    multiprocessing = experimental_setup['multiprocessing']
    n_processes = experimental_setup['n_processes']

    if multiprocessing:
        with MultiprocessingEvaluator(model, n_processes=n_processes) as evaluator:
            results = evaluator.perform_experiments(
                scenarios=n_scenarios, policies=n_policies)
    else:
        results = perform_experiments(
            models=model, scenarios=n_scenarios, policies=n_policies)

    save_experiment_results(country, return_period, model,
                            results, n_scenarios, n_policies)


def save_experiment_results(country: str, return_period: int, model: Model, results, n_scenarios, n_policies):
    """Saves experiment results to a file, taking into account if there was a conflict."""
    results_path = Path(f'../experiments/{country}')
    results_path.mkdir(parents=True, exist_ok=True)

    is_conflict = getattr(model.constants._data.get(
        'is_conflict'), 'value', False)

    conflict_str = ", conflict=True" if is_conflict else ""
    filename = f"return_period={return_period}, scenarios={n_scenarios}, policies={n_policies}{conflict_str}.tar.gz"
    save_results(results, results_path / filename)
