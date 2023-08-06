# Prepare the results of the experiments for the analysis.

import pandas as pd
import numpy as np
import ast
import geopandas as gpd
from ema_workbench import load_results


def prepare_outcomes(results: tuple, add_policies: bool, add_uncertainties: bool) -> pd.DataFrame:
    '''Convert outcomes dict into a data frame.

    Args:
        results (tuple): The results of the experiments in the EMA Workbench format.
        add_policies (bool): Whether to add policy values to the data frame.
        add_uncertainties (bool): Whether to add uncertainty values to the data frame.

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

    uncertainty_names = ['consumption_utility',
                         'discount_rate',
                         'income_and_expenditure_growth',
                         'poverty_bias']

    experiments, _ = results
    experiments['random_seed'] = experiments['random_seed'].astype(int)
    experiments['scenario'] = experiments['scenario'].astype(int)
    if len(experiments['random_seed'].unique()) != experiments['scenario'].max() - experiments['scenario'].min() + 1:
        # print(experiments['random_seed'].value_counts())
        print('WARNING! Random seeds are not unique.')
        # raise ValueError('Random seeds are not unique')

    policy_names = ['my_policy']

    if add_policies:
        if add_uncertainties:
            columns = ['scenario', 'policy', 'district', 'random_seed'] + \
                policy_names + uncertainty_names + outcome_names
        else:
            columns = ['scenario', 'policy', 'district', 'random_seed'] + \
                policy_names + outcome_names
    else:
        if add_uncertainties:
            columns = ['scenario', 'policy', 'district', 'random_seed'] + \
                uncertainty_names + outcome_names
        else:
            columns = ['scenario', 'policy', 'district',
                       'random_seed'] + outcome_names

    scenarios = results[0]['scenario'].values
    n_scenarios = results[0]['scenario'].unique().size
    policies = results[0]['policy'].values
    random_seeds = results[0]['random_seed'].values
    n_policies = results[0]['policy'].unique().size
    n_districts = len(results[1].keys())

    if add_policies:
        policy_values = results[0][policy_names].values

    if add_uncertainties:
        uncertainty_values = results[0][uncertainty_names].values

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
            outcomes[i, 3] = random_seeds[k]

            if add_policies:
                if add_uncertainties:
                    # Add policy values
                    # From 4 to 4 + len(policy_names) policy values
                    for j, name in enumerate(policy_names):
                        outcomes[i, 4 + j] = policy_values[k, j]

                    # Add uncertainty values
                    # From 4 + len(policy_names) to 4 + len(policy_names) + len(uncertainty_names) uncertainty values
                    for j, name in enumerate(uncertainty_names):
                        outcomes[i, 4 + len(policy_names) + j] = uncertainty_values[k, j]

                    # Add outcomes
                    # From 4 + len(policy_names) + len(uncertainty_names) to 4 + len(policy_names) + len(uncertainty_names) + len(outcome_names) outcomes
                    l = 4 + len(policy_names) + len(uncertainty_names)
                    for v, name in zip(arr, outcome_names):
                        if name == 'years_in_poverty':
                            outcomes[i, l] = ast.literal_eval(v)
                        else:
                            outcomes[i, l] = v
                        l += 1
                else:
                    # Add policy values
                    # From 4 to 4 + len(policy_names) policy values
                    for j, name in enumerate(policy_names):
                        outcomes[i, 4 + j] = policy_values[k, j]

                    # Add outcomes
                    # From 4 + len(policy_names) to 4 + len(policy_names) + len(outcome_names) outcomes
                    l = 4 + len(policy_names)
                    for v, name in zip(arr, outcome_names):
                        if name == 'years_in_poverty':
                            outcomes[i, l] = ast.literal_eval(v)
                        else:
                            outcomes[i, l] = v
                        l += 1
            else:
                if add_uncertainties:
                    # Add uncertainty values
                    # From 4 to 4 + len(uncertainty_names) uncertainty values
                    for j, name in enumerate(uncertainty_names):
                        outcomes[i, 4 + j] = uncertainty_values[k, j]

                    # Add outcomes
                    # From 4 + len(uncertainty_names) to 4 + len(uncertainty_names) + len(outcome_names) outcomes
                    l = 4 + len(uncertainty_names)
                    for v, name in zip(arr, outcome_names):
                        if name == 'years_in_poverty':
                            outcomes[i, l] = ast.literal_eval(v)
                        else:
                            outcomes[i, l] = v
                        l += 1
                else:
                    # Add outcomes
                    # From 4 to 4 + len(outcome_names) outcomes
                    l = 4
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
    if add_policies:
        numeric_columns = outcomes.columns[4:-1].tolist()
        outcomes[numeric_columns] = outcomes[numeric_columns].apply(
            pd.to_numeric)
    else:
        numeric_columns = outcomes.columns[3:-1].tolist()
        outcomes[numeric_columns] = outcomes[numeric_columns].apply(
            pd.to_numeric)

    # Rename a district
    outcomes['district'].replace(
        {'AnseLaRayeCanaries': 'Anse-La-Raye & Canaries'}, inplace=True)

    # Convert pct columns to percentage
    outcomes['annual_average_consumption_loss_pct'] = outcomes['annual_average_consumption_loss_pct'] * 100
    outcomes['initial_poverty_gap'] = outcomes['initial_poverty_gap'] * 100
    outcomes['new_poverty_gap'] = outcomes['new_poverty_gap'] * 100

    # Calculate the percentage of new poor
    # outcomes = outcomes.assign(n_new_poor_increase_pct=outcomes['n_new_poor'].div(
    #     outcomes['total_population']).multiply(100))

    outcomes['pct_poor_before'] = outcomes['n_poor_initial'].div(
        outcomes['total_population'])
    outcomes['pct_poor_after'] = outcomes['n_new_poor'].add(
        outcomes['n_poor_initial']).div(outcomes['total_population'])
    outcomes['pct_poor_increase'] = outcomes['pct_poor_after'].sub(
        outcomes['pct_poor_before'])

    # Move years_in_poverty column to the end of the data frame
    outcomes = outcomes[[c for c in outcomes if c not in [
        'years_in_poverty']] + ['years_in_poverty']]

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


def get_policy_effectiveness_tab(outcomes: pd.DataFrame) -> pd.DataFrame:
    policy_name_mapper = {'all+0': 'None',
                          'all+10': '10% to all',
                          'all+30': '30% to all',
                          'all+50': '50% to all',
                          'all+100': '100% to all',
                          'poor+0': 'None',
                          'poor+10': '10% to poor',
                          'poor+30': '30% to poor',
                          'poor+50': '50% to poor',
                          'poor+100': '100% to poor',
                          'poor_near_poor1.25+0': 'None',
                          'poor_near_poor1.25+10': '10% to poor and near poor (1.25)',
                          'poor_near_poor1.25+30': '30% to poor and near poor (1.25)',
                          'poor_near_poor1.25+50': '50% to poor and near poor (1.25)',
                          'poor_near_poor1.25+100': '100% to poor and near poor (1.25)',
                          'poor_near_poor2.0+0': 'None',
                          'poor_near_poor2.0+10': '10% to poor and near poor (2.0)',
                          'poor_near_poor2.0+30': '30% to poor and near poor (2.0)',
                          'poor_near_poor2.0+50': '50% to poor and near poor (2.0)',
                          'poor_near_poor2.0+100': '100% to poor and near poor (2.0)'}
    df = outcomes.copy()
    df['my_policy'] = df['my_policy'].replace(policy_name_mapper)
    df['my_policy'] = pd.Categorical(df['my_policy'], categories=['None', '10% to all', '30% to all', '50% to all', '100% to all',
                                                                  '10% to poor', '30% to poor', '50% to poor', '100% to poor',
                                                                  '10% to poor and near poor (1.25)', '30% to poor and near poor (1.25)', '50% to poor and near poor (1.25)', '100% to poor and near poor (1.25)',
                                                                  '10% to poor and near poor (2.0)', '30% to poor and near poor (2.0)', '50% to poor and near poor (2.0)', '100% to poor and near poor (2.0)'], ordered=True)
    df.rename(columns={'my_policy': 'Policy',
                       'district': 'District'}, inplace=True)
    df.rename(columns={'annual_average_consumption_loss_pct': 'Annual average consumption loss (%)',
                       'n_new_poor': 'Number of new poor'},
              inplace=True)
    df['Policy ID'] = df['Policy'].cat.codes
    return df
