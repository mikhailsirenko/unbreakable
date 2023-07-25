import pandas as pd
import numpy as np
import ast
from ema_workbench import load_results


def prepare_outcomes(results: tuple, add_policies: bool) -> pd.DataFrame:
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
        'poverty_line_adjusted',
        'pml',
        'n_poor_initial',
        'n_poor_affected',
        'n_new_poor',
        'initial_poverty_gap',
        'new_poverty_gap',
        'annual_average_consumption_loss',
        'annual_average_consumption_loss_pct',
        'r',
        'years_in_poverty'
    ]
    experiments, _ = results
    experiments['random_seed'] = experiments['random_seed'].astype(int)
    experiments['scenario'] = experiments['scenario'].astype(int)
    if len(experiments['random_seed'].unique()) != experiments['scenario'].max() - experiments['scenario'].min() + 1:
        raise ValueError('Random seeds are not unique')

    policy_names = ['my_policy']

    if add_policies:
        columns = ['scenario', 'policy', 'district'] + \
            policy_names + outcome_names
    else:
        columns = ['scenario', 'policy', 'district'] + outcome_names

    scenarios = results[0]['scenario'].values
    n_scenarios = results[0]['scenario'].unique().size
    policies = results[0]['policy'].values
    n_policies = results[0]['policy'].unique().size
    n_districts = len(results[1].keys())

    if add_policies:
        policy_values = results[0][policy_names].values

    n_columns = len(columns)
    n_rows = n_scenarios * n_policies * n_districts
    outcomes = np.zeros((n_rows, n_columns), dtype=object)

    i = 0  # to iterate over rows = scenarios * policies * districts
    for district, district_outcomes in results[1].items():
        # to iterate over rows = scenarios * policies (experiments dataframe)
        k = 0
        # We reset k every time we change district
        for arr in district_outcomes:
            # The first 3 rows for scenario, policy and district
            outcomes[i, 0] = scenarios[k]
            outcomes[i, 1] = policies[k]
            outcomes[i, 2] = district

            if add_policies:
                # Add policy values
                # From 3 to 3 + len(policy_names) policy values
                for j, name in enumerate(policy_names):
                    outcomes[i, 3 + j] = policy_values[k, j]

                # Add outcomes
                # From 3 + len(policy_names) to 3 + len(policy_names) + len(outcome_names) outcomes
                l = 3 + len(policy_names)
                for v, name in zip(arr, outcome_names):
                    if name == 'years_in_poverty':
                        outcomes[i, l] = ast.literal_eval(v)
                    else:
                        outcomes[i, l] = v
                    l += 1
            k += 1  # increase row index to get next experiment for the current district
            i += 1  # increase row index of the outcomes dataframe

    return pd.DataFrame(outcomes, columns=columns)