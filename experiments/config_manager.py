import yaml
from pathlib import Path


def load_config(country: str, return_period: int, disaster_type: str, is_conflict: bool = False) -> dict:
    '''Load configuration for the specified case country.
    
    Args:
        country (str): The country for which to load the configuration.
        return_period (int): The return period for the disaster.
        disaster_type (str): The type of disaster.
        is_conflict (bool): Whether the country is in conflict.
    
    Returns:
        dict: The configuration for the specified case country.
    '''

    config_path = Path(f"../config/{country}.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file for {country} not found at {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    check_config_parameters(config)

    config['constants']['return_period'] = return_period
    config['constants']['disaster_type'] = disaster_type

    if is_conflict:
        config['constants']['is_conflict'] = True
    else:
        config['constants']['is_conflict'] = False

    return config

def check_config_parameters(config: dict) -> None:
    '''Check if the configuration parameters are valid.
    
    Args:
        config (dict): The configuration to check.
    
    Returns:
        None
    
    Raises:
        ValueError: If the configuration parameters are not valid.
    '''
    return_periods = [10, 50, 100, 250, 500, 1000]
    disaster_types = ['hurricane', 'flood']

    if 'return_period' not in config['constants']:
        raise ValueError("Return period not specified in configuration.")

    if 'disaster_type' not in config['constants']:
        raise ValueError("Disaster type not specified in configuration.")

    if 'return_period' not in return_periods:
        raise ValueError(
            f"Return period {config['constants']['return_period']} not in available return periods: {return_periods}")

    if 'disaster_type' not in disaster_types:
        raise ValueError(
            f"Disaster type {config['constants']['disaster_type']} not in available disaster types: ['hurricane', 'flood']")

    neccessary_parameters = ['country', 'avg_prod', 'inc_exp_growth', 'cons_util', 'disc_rate', 'disaster_type', 'calc_exposure_params', 'identify_aff_params', 'add_inc_loss', 'pov_bias', 'lambda_incr', 'yrs_to_rec', 'rnd_inc_params', 'rnd_sav_params', 'rnd_rent_params', 'rnd_house_vuln_params', 'min_households', 'atol', 'save_households', 'save_consumption_recovery', 'regions', 'levers', 'uncertainties']
    exposure_neccessary_parameters = ['distr', 'high', 'low']
    identify_aff_neccessary_parameters = ['delta_pct', 'distr', 'high', 'low', 'num_masks']
    rnd_inc_neccessary_parameters = ['randomize', 'distr', 'delta']
    rnd_sav_neccessary_parameters = ['randomize', 'distr', 'avg', 'delta']
    rnd_rent_neccessary_parameters = ['randomize', 'distr', 'avg', 'delta']
    rnd_house_vuln_neccessary_parameters = ['randomize', 'distr', 'low', 'high', 'min_thresh', 'max_thresh']

    for parameter in neccessary_parameters:
        if parameter not in config['constants']:
            raise ValueError(f"Parameter {parameter} not found in configuration.")
    
    for parameter in exposure_neccessary_parameters:
        if parameter not in config['constants']['calc_exposure_params']:
            raise ValueError(f"Parameter {parameter} not found in calc_exposure_params.")
        
    for parameter in identify_aff_neccessary_parameters:
        if parameter not in config['constants']['identify_aff_params']:
            raise ValueError(f"Parameter {parameter} not found in identify_aff_params.")

    for parameter in rnd_inc_neccessary_parameters:
        if parameter not in config['constants']['rnd_inc_params']:
            raise ValueError(f"Parameter {parameter} not found in rnd_inc_params.")
    
    for parameter in rnd_sav_neccessary_parameters:
        if parameter not in config['constants']['rnd_sav_params']:
            raise ValueError(f"Parameter {parameter} not found in rnd_sav_params.")
    
    for parameter in rnd_rent_neccessary_parameters:
        if parameter not in config['constants']['rnd_rent_params']:
            raise ValueError(f"Parameter {parameter} not found in rnd_rent_params.")
        
    for parameter in rnd_house_vuln_neccessary_parameters:
        if parameter not in config['constants']['rnd_house_vuln_params']:
            raise ValueError(f"Parameter {parameter} not found in rnd_house_vuln_params.")