# Prepare the results of the experiments for the analysis.

import pandas as pd
import numpy as np
import ast
import geopandas as gpd
from ema_workbench import load_results


def prepare_outcomes(results: tuple, add_policies: bool) -> pd.DataFrame:
    '''Convert outcomes dict into a data frame.

    Args:
        results (tuple): The results of the experiments in the EMA Workbench format.

    Returns:
        pd.DataFrame: Outcomes data frame.
    '''
    # * Note that we specify all outcomes in `get_outcomes` function in `write.py`
    # * Here we just read them in the same sequence that they are written
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
        'recovery_rate',
        'years_in_poverty',
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

    i = 0  # To iterate over rows = scenarios * policies * districts
    for district, district_outcomes in results[1].items():
        # To iterate over rows = scenarios * policies (experiments dataframe)
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
            else:
                # Add outcomes
                # From 3 to 3 + len(outcome_names) outcomes
                l = 3
                for v, name in zip(arr, outcome_names):
                    if name == 'years_in_poverty':
                        outcomes[i, l] = ast.literal_eval(v)
                    else:
                        outcomes[i, l] = v
                    l += 1
            k += 1  # increase row index to get next experiment for the current district
            i += 1  # increase row index of the outcomes dataframe
    outcomes = pd.DataFrame(outcomes, columns=columns)

    # Convert numeric columns to numeric
    numeric_columns = outcomes.columns[3:-1].tolist()
    outcomes[numeric_columns] = outcomes[numeric_columns].apply(pd.to_numeric)

    # Rename a district
    outcomes['district'].replace(
        {'AnseLaRayeCanaries': 'Anse-La-Raye & Canaries'}, inplace=True)

    # Convert pct columns to percentage
    outcomes['annual_average_consumption_loss_pct'] = outcomes['annual_average_consumption_loss_pct'] * 100
    outcomes['initial_poverty_gap'] = outcomes['initial_poverty_gap'] * 100
    outcomes['new_poverty_gap'] = outcomes['new_poverty_gap'] * 100

    # Calculate the percentage of new poor
    outcomes = outcomes.assign(n_new_poor_increase_pct=outcomes['n_new_poor'].div(
        outcomes['total_population']).multiply(100))

    return outcomes


def get_spatial_outcomes(outcomes: pd.DataFrame, outcomes_of_interest: list = [], country: str = 'Saint Lucia', aggregation: str = 'mean') -> gpd.GeoDataFrame:
    # Load country shapefile
    country = 'Saint Lucia'
    gdf = gpd.read_file(
        f'../data/raw/shapefiles/{country}/gadm36_LCA_shp/gadm36_LCA_1.shp')

    # Align district names with the ones in the outcomes
    gdf['NAME_1'].replace(
        {'Soufrière': 'Soufriere', 'Vieux Fort': 'Vieuxfort'}, inplace=True)

    # Merge Anse-la-Raye and Canaries into a single geometry
    geometry = gdf[gdf['NAME_1'].isin(
        ['Anse-la-Raye', 'Canaries'])].unary_union

    # Add it to the dataframe
    gdf.loc[len(gdf)] = [None, None, 'LCA.11_1', 'Anse-La-Raye & Canaries',
                         None, None, None, None, None, None, geometry]
    gdf = gdf[gdf['NAME_1'].isin(outcomes['district'].unique())]

    if len(outcomes_of_interest) == 0:
        outcomes_of_interest = ['total_asset_loss',
                                'total_consumption_loss',
                                'n_affected_people',
                                'n_new_poor',
                                'new_poverty_gap',
                                'annual_average_consumption_loss',
                                'annual_average_consumption_loss_pct',
                                'n_new_poor_increase_pct',
                                'r']

    # Aggregate outcomes
    if aggregation == 'mean':
        aggregated = outcomes[['district'] +
                              outcomes_of_interest].groupby('district').mean()
    elif aggregation == 'median':
        aggregated = outcomes[['district'] +
                              outcomes_of_interest].groupby('district').median()
    else:
        raise ValueError('Aggregation must be either mean or median')

    # Merge with the shapefile
    gdf = pd.merge(gdf, aggregated, left_on='NAME_1', right_index=True)
    gdf.reset_index(inplace=True, drop=True)
    return gdf
