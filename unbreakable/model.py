import numpy as np
from ema_workbench import *
from unbreakable.data.saver import *
from unbreakable.data.reader import *
from unbreakable.data.randomizer import *
from unbreakable.analysis.calculator import *
from unbreakable.modules.disaster import *
from unbreakable.modules.households import *
from unbreakable.modules.policy import *
from unbreakable.modules.recovery import *


def model(**params) -> dict:
    '''Simulation model.

    Args:
        params (dict): A dictionary of parameters. For a complete list of parameters, see, e.g., `config/Nigeria.yaml`.

    Returns:
        dict: A dictionary of outcomes. The key is a district and value is a list of outcomes.
    '''
    random_seed = params['random_seed']

    households, risk_and_damage, conflict = read_data(
        params['country'], params['is_conflict'])

    households = randomize(
        households, risk_and_damage, params, random_seed)

    # ! What is this?
    welfare = calculate_welfare(households, params['cons_util'])

    # Store outcomes in a dict, where key is a region and value is a list of outcomes
    outcomes = {}

    for region in params['regions']:
        region_households = households[households['region'] == region].copy(
        )

        total_exposed_stock, expected_loss_fraction, region_pml = get_region_damage(
            risk_and_damage, region, params['return_period'])

        region_households = (region_households
                             .pipe(calculate_exposure, region_pml, params['pov_bias'], params['calc_exposure_params'])
                             .pipe(identify_affected, region_pml, params['identify_aff_params'], random_seed=random_seed)
                             .pipe(apply_policy, params['country'], params.get('current_policy', 'none'), params['disaster_type'], random_seed=random_seed)
                             .pipe(calculate_recovery_rate, params['avg_prod'], params['cons_util'], params['disc_rate'], params['lambda_incr'], params['yrs_to_rec'], params['is_conflict'], conflict)
                             .pipe(calculate_wellbeing, params['avg_prod'], params['cons_util'], params['disc_rate'], params['inc_exp_growth'], params['yrs_to_rec'], params['add_inc_loss'], params['save_consumption_recovery'], params['is_conflict']))

        if params['save_households']:
            save_households(
                params['country'], region, region_households, random_seed)

        outcomes[region] = np.array(list(calculate_outcomes(
            region_households, total_exposed_stock, expected_loss_fraction, region_pml, params['yrs_to_rec'], welfare).values()))

    return outcomes
