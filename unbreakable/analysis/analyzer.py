"""This module contains functions to prepare the results of the experiments for analysis."""

import pandas as pd
import numpy as np
import ast
import geopandas as gpd
from tqdm import tqdm
import yaml


def prepare_outcomes(results: tuple, add_policies: bool, add_uncertainties: bool) -> pd.DataFrame:
    '''Convert outcomes dict in (EMA Workbench format) into a pd.DataFrame.

    Args:
        results (tuple): The results of the experiments in the EMA Workbench format.
        add_policies (bool): Whether to add policy values.
        add_uncertainties (bool): Whether to add uncertainty values.

    Returns:
        pd.DataFrame: Outcomes.
    '''
    # Read outcome names from a yaml file
    with open("../../unbreakable/analysis/outcomes.yaml", "r") as f:
        outcome_names = yaml.safe_load(f)

    # TODO: Read uncertainty names from results
    # uncertainty_names = ['consump_util',
    #                      'discount_rate',
    #                      'income_and_expenditure_growth',
    #                      'poverty_bias']
    uncertainty_names = []

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
                        outcomes[i, 4 + len(policy_names) +
                                 j] = uncertainty_values[k, j]

                    # Add outcomes
                    # From 4 + len(policy_names) + len(uncertainty_names) to 4 + len(policy_names) + len(uncertainty_names) + len(outcome_names) outcomes
                    l = 4 + len(policy_names) + len(uncertainty_names)
                    for v, name in zip(arr, outcome_names):
                        if name in ['weighted_vuln_quint', 'weighted_vuln_dec', 'years_in_poverty']:
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
                        if name in ['weighted_vuln_quint', 'weighted_vuln_dec', 'years_in_poverty']:
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
                        if name in ['weighted_vuln_quint', 'weighted_vuln_dec', 'years_in_poverty']:
                            outcomes[i, l] = ast.literal_eval(v)
                        else:
                            outcomes[i, l] = v
                        l += 1
                else:
                    # Add outcomes
                    # From 4 to 4 + len(outcome_names) outcomes
                    l = 4
                    for v, name in zip(arr, outcome_names):
                        if name in ['weighted_vuln_quint', 'weighted_vuln_dec', 'years_in_poverty']:
                            outcomes[i, l] = ast.literal_eval(v)
                        else:
                            outcomes[i, l] = v
                        l += 1
            k += 1  # increase row index to get next experiment for the current district
            i += 1  # increase row index of the outcomes dataframe
    outcomes = pd.DataFrame(outcomes, columns=columns)

    numeric_columns = ['total_population',
                       'total_asset_loss',
                       'total_consumption_loss',
                       'tot_exposed_asset',
                       'tot_asset_surv',
                       'expected_loss_frac',
                       'n_affected_people',
                       'annual_average_consumption',
                       'poverty_line_adjusted',
                       'district_pml',
                       'n_poor_initial',
                       'n_poor_affected',
                       'n_new_poor',
                       'initial_poverty_gap',
                       'new_poverty_gap_initial',
                       'new_poverty_gap_all',
                       'annual_average_consumption_loss',
                       'annual_average_consumption_loss_pct',
                       'mean_recovery_rate',
                       'r']

    outcomes[numeric_columns] = outcomes[numeric_columns].apply(pd.to_numeric)

    # Rename a district
    outcomes['district'].replace(
        {'AnseLaRayeCanaries': 'Anse-La-Raye & Canaries'}, inplace=True)

    # Convert pct columns to percentage
    outcomes['annual_average_consumption_loss_pct'] = outcomes['annual_average_consumption_loss_pct'] * 100
    outcomes['initial_poverty_gap'] = outcomes['initial_poverty_gap'] * 100
    outcomes['new_poverty_gap_all'] = outcomes['new_poverty_gap_all'] * 100
    outcomes['new_poverty_gap_initial'] = outcomes['new_poverty_gap_initial'] * 100

    # Calculate the percentage of new poor
    outcomes = outcomes.assign(n_new_poor_increase_pp=outcomes['n_new_poor'].div(
        outcomes['total_population']).multiply(100))

    # Move years_in_poverty column to the end of the data frame
    outcomes = outcomes[[c for c in outcomes if c not in [
        'years_in_poverty']] + ['years_in_poverty']]

    return outcomes


def get_spatial_outcomes(outcomes: pd.DataFrame, outcomes_of_interest: list = [], country: str = 'Saint Lucia', aggregation: str = 'mean') -> gpd.GeoDataFrame:
    '''Connect outcomes of interest with the shapefile.

    Args:
        outcomes (pd.DataFrame): Outcomes.
        outcomes_of_interest (list, optional): Outcomes of interest. Defaults to [].
        country (str, optional): Country name. Defaults to 'Saint Lucia'.
        aggregation (str, optional): Aggregation method. Defaults to 'mean'.

    Returns:
        gpd.GeoDataFrame: Spatial outcomes.
    '''

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
                                'tot_exposed_asset',
                                'total_consumption_loss',
                                'n_affected_people',
                                'n_new_poor',
                                'new_poverty_gap_all',
                                'annual_average_consumption_loss',
                                'annual_average_consumption_loss_pct',
                                'n_new_poor_increase_pp',
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
    '''Get the policy effectiveness table.

    Effectiveness here is how effective the policy is in respect to:
    - reducing the number of new poor;
    - reducing the average consumption loss.

    Args:
        outcomes (pd.DataFrame): Outcomes.

    Returns:
        pd.DataFrame: Policy effectiveness table.
    '''
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


def get_weeks_in_poverty_tab(outcomes: pd.DataFrame) -> pd.DataFrame:
    '''Get the average across scenarios number of weeks in poverty for each district.

    Args:
        outcomes (pd.DataFrame): Outcomes.

    Returns:
        pd.DataFrame: Average number of weeks in poverty for each district.
    '''
    # Specify the districts
    districts = ['Anse-La-Raye & Canaries', 'Castries', 'Choiseul',
                 'Dennery', 'Gros Islet', 'Laborie', 'Micoud', 'Soufriere', 'Vieuxfort']

    # Keep track of the averages
    district_average = {}

    # Get the number of scenarios
    n_scenarios = outcomes['scenario'].unique().size

    # Get the keys
    column_name = 'years_in_poverty'
    all_keys = outcomes[column_name][0].keys()

    # Iterate through the districts
    for district in districts:
        # Subset the outcomes for a specific district
        df = outcomes[outcomes['district'] == district]

        # Get the dictionaries from the column
        dicts = df[column_name].tolist()

        # Initialize the sums
        sums = dict(zip(all_keys, [0] * len(all_keys)))

        # Iterate through the dictionaries and update the sums
        for d in dicts:
            for key in all_keys:
                if key in d:
                    sums[key] += d[key]
                else:
                    sums[key] += 0

        # Calculate the average
        district_average[district] = {
            key: sums[key] / n_scenarios for key in all_keys}

    # Convert the dictionary to a dataframe
    result = pd.DataFrame(district_average).T
    result.index.name = 'District'
    result.columns = [i for i in range(0, len(all_keys))]
    # result.columns = [int(x) if int(x) < len(
    #     all_keys) else f'>{len(all_keys)}' for x in range(1, len(all_keys) + 1)]
    return result


def get_average_weighted_vulnerability(outcomes: pd.DataFrame, quintile: bool) -> pd.DataFrame:
    '''Get the average weighted vulnerability for each district.

    Args:
        outcomes (pd.DataFrame): Outcomes.
        quintile (bool): Whether to calculate the average weighted vulnerability by quintile or decile.

    Returns:
        pd.DataFrame: Average weighted vulnerability for each district.
    '''
    # Specify the districts
    districts = ['Anse-La-Raye & Canaries', 'Castries', 'Choiseul',
                 'Dennery', 'Gros Islet', 'Laborie', 'Micoud', 'Soufriere', 'Vieuxfort']

    # Keep track of the averages
    district_average = {}

    # Get the number of scenarios
    n_scenarios = outcomes['scenario'].unique().size

    if quintile:
        column_name = 'weighted_vuln_quint'
        all_keys = [1, 2, 3, 4, 5]
    else:
        column_name = 'weighted_vuln_dec'
        all_keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Iterate through the districts
    for district in districts:
        # Subset the outcomes for a specific district
        df = outcomes[outcomes['district'] == district]

        # Get the dictionaries from the column
        dicts = df[column_name].tolist()

        # Initialize the sums
        sums = dict(zip(all_keys, [0] * len(all_keys)))

        # Iterate through the dictionaries and update the sums
        for d in dicts:
            for key in all_keys:
                if key in d:
                    sums[key] += d[key]
                else:
                    sums[key] += 0

        # Calculate the average
        district_average[district] = {
            key: sums[key] / n_scenarios for key in all_keys}

    # Convert the dictionary to a dataframe
    result = pd.DataFrame(district_average).T

    # Rename the index and columns
    result.index.name = 'District'
    if quintile:
        result.columns.name = 'Quintile'
    else:
        result.columns.name = 'Decile'
    return result


def calculate_resilience(affected_households: pd.DataFrame, tot_wellbeing_loss: float) -> float:
    '''Calculate socio-economic resilience of affected households.

    Socio-economic resilience is a ratio of asset loss to consumption loss.

    Args:
        affected_households (pd.DataFrame): Affected households.
        tot_wellbeing_loss (float): Total wellbeing loss.

    Returns:
        float: Socio-economic resilience
    '''
    total_consumption_loss = (
        affected_households[['consumption_loss_NPV', 'popwgt']].prod(axis=1)).sum()

    total_asset_damage = (
        affected_households[['keff', 'v', 'popwgt']].prod(axis=1)).sum()

    if total_consumption_loss == 0:
        r = np.nan

    else:
        r = total_asset_damage / total_consumption_loss
        # r = total_asset_damage / tot_wellbeing_loss

    return r
