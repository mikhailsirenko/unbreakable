import pandas as pd
import numpy as np


def prepare_outcomes_dict(results: pd.DataFrame, district: str, add_scenario_column: bool, add_policy_column: bool, used_policies: list = ['None']) -> dict:
    '''Prepare outcomes for a single `district`.'''
    outcome_names = ['total_population',
                     'n_affected_households',
                     'annual_average_consumption',
                     'poverty_line',
                     'pml',
                     'n_poor_initial',
                     'n_new_poor',
                     'initial_poverty_gap',
                     'new_poverty_gap',
                     'annual_average_consumption_loss',
                     'annual_average_consumption_loss_pct',
                     'r']

    outcomes = {}

    for i, outcome_name in enumerate(outcome_names):
        l = []
        for arr in results[1][district]:
            l.append(arr[i])
        outcomes[outcome_name] = l

    used_scenarios = results[0]['scenario'].unique().sort_values().tolist()
    scenarios = []
    policies = []

    for _, policy in enumerate(used_policies):
        for _, scenario in enumerate(used_scenarios):
            if add_scenario_column:
                scenarios.append(scenario)
            if add_policy_column:
                policies.append(policy)
    if add_scenario_column:
        outcomes[f'scenario'] = scenarios
    if add_policy_column:
        outcomes[f'policy'] = policies

    return outcomes


def prepare_outcomes_dataframe(results) -> pd.DataFrame:
    '''Convert outcomes dict into a data frame.

    Args:
        results (tuple): The results of the experiments.

    Returns:
        pd.DataFrame: Outcomes data frame.
    '''

    outcome_names = [
                     'total_population',
                     'total_asset_loss',
                     'total_consumption_loss',
                     'event_damage',
                     'total_asset_stock',
                     'average_productivity',
                     'total_asset_in_survey',
                     'expected_loss_fraction',
                     'n_affected_people',
                     'annual_average_consumption',
                     'poverty_line',
                     'pml',
                     'n_poor_initial',
                     'n_new_poor',
                     'initial_poverty_gap',
                     'new_poverty_gap',
                     'annual_average_consumption_loss',
                     'annual_average_consumption_loss_pct',
                     'r'
                     ]

    columns = ['scenario', 'policy', 'district'] + outcome_names
    
    scenarios = results[0]['scenario'].unique()
    n_scenarios = scenarios.size
    policies = results[0]['policy'].unique()
    n_policies = policies.size
    n_districts = len(results[1].keys())

    n_columns = len(columns)
    n_rows = n_scenarios * n_policies * n_districts

    outcomes = np.zeros((n_rows, n_columns), dtype=object)

    k = 0  # row index
    for district, district_outcomes in results[1].items():
        i = 0  # scenario index
        j = 0  # policy index
        for arr in district_outcomes:
            if i == n_scenarios:
                i = 0
                j += 1

            outcomes[k, 0] = scenarios[i]
            outcomes[k, 1] = policies[j]
            outcomes[k, 2] = district

            l = 3
            for v, name in zip(arr, outcome_names):
                outcomes[k, l] = v
                l += 1

            k += 1
            i += 1

    return pd.DataFrame(outcomes, columns=columns)
