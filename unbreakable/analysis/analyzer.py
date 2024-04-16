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

    policy_names = ['current_policy']
    uncertainty_names = []

    experiments, _ = results
    experiments['random_seed'] = experiments['random_seed'].astype(int)
    experiments['scenario'] = experiments['scenario'].astype(int)

    if len(experiments['random_seed'].unique()) != experiments['scenario'].max() - experiments['scenario'].min() + 1:
        print('Warning! Random seeds are not unique.')

    base_columns = ['scenario', 'policy', 'region', 'random_seed']
    if add_policies:
        if add_uncertainties:
            columns = base_columns + \
                policy_names + uncertainty_names + outcome_names
        else:
            columns = base_columns + \
                policy_names + outcome_names
    else:
        if add_uncertainties:
            columns = base_columns + \
                uncertainty_names + outcome_names
        else:
            columns = base_columns + outcome_names

    scenarios = results[0]['scenario'].values
    n_scenarios = results[0]['scenario'].unique().size

    policies = results[0]['policy'].values
    n_policies = results[0]['policy'].unique().size

    random_seeds = results[0]['random_seed'].values
    n_regions = len(results[1].keys())

    if add_policies:
        policy_values = results[0][policy_names].values

    if add_uncertainties:
        uncertainty_values = results[0][uncertainty_names].values

    n_columns = len(columns)
    n_rows = n_scenarios * n_policies * n_regions
    outcomes = np.zeros((n_rows, n_columns), dtype=object)

    i = 0  # To iterate over rows = scenarios * policies * regions
    for region, region_outcomes in results[1].items():
        # To iterate over rows = scenarios * policies (experiments dataframe)
        k = 0
        # We reset k every time we change region
        for arr in region_outcomes:
            # The first 3 rows for scenario, policy and region
            outcomes[i, 0] = scenarios[k]
            outcomes[i, 1] = policies[k]
            outcomes[i, 2] = region
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
            k += 1  # increase row index to get next experiment for the current region
            i += 1  # increase row index of the outcomes dataframe
    outcomes = pd.DataFrame(outcomes, columns=columns)

    # Iterate over the columns and try to convert them to numeric if possible
    for col in outcomes.columns:
        try:
            outcomes[col] = pd.to_numeric(outcomes[col])
        except:
            pass

    # Convert pct columns to percentage
    outcomes['annual_avg_consum_loss_pct'] = outcomes['annual_avg_consum_loss_pct'] * 100
    outcomes['initial_poverty_gap'] = outcomes['initial_poverty_gap'] * 100
    outcomes['new_poverty_gap_all'] = outcomes['new_poverty_gap_all'] * 100
    outcomes['new_poverty_gap_initial'] = outcomes['new_poverty_gap_initial'] * 100
    outcomes['n_poor_ratio'] = outcomes['n_poor_initial'].div(
        outcomes['tot_pop']).round(2) * 100

    # Calculate the percentage of new poor
    outcomes = outcomes.assign(n_new_poor_increase_pp=outcomes['n_new_poor'].div(
        outcomes['tot_pop']).multiply(100))

    # Move years_in_poverty column to the end of the data frame
    outcomes = outcomes[[c for c in outcomes if c not in [
        'years_in_poverty']] + ['years_in_poverty']]

    return outcomes


def get_spatial_outcomes(outcomes: pd.DataFrame, country: str, outcomes_of_interest: list = [], aggregation: str = 'median') -> gpd.GeoDataFrame:
    '''Connect outcomes with spatial data.

    Args:
        outcomes (pd.DataFrame): Outcomes data frame.
        country (str, optional): Country name.
        outcomes_of_interest (list, optional): List of outcomes of interest. Defaults to [].
        aggregation (str, optional): Aggregation method. Defaults to 'median'.

    Returns:
        gpd.GeoDataFrame: Spatial outcomes.
    '''

    if aggregation not in ['mean', 'median']:
        raise ValueError('Aggregation must be either mean or median')

    if country == 'Saint Lucia':
        column = 'NAME_1'
        gdf = gpd.read_file(
            f'../data/raw/shapefiles/{country}/gadm36_LCA_shp/gadm36_LCA_1.shp')

        gdf['NAME_1'].replace(
            {'Soufrière': 'Soufriere', 'Vieux Fort': 'Vieuxfort'}, inplace=True)

        # Merge Anse-la-Raye and Canaries into a single geometry
        geometry = gdf[gdf['NAME_1'].isin(
            ['Anse-la-Raye', 'Canaries'])].unary_union

        # Add it to the dataframe
        gdf.loc[len(gdf)] = [None, None, 'LCA.11_1', 'Anse-La-Raye & Canaries',
                             None, None, None, None, None, None, geometry]
        gdf = gdf[gdf['NAME_1'].isin(outcomes['region'].unique())]

    elif country == 'Dominica':
        column = 'NAME'
        gdf = gpd.read_file(
            '../../data/raw/shapefiles/Dominica/dma_admn_adm1_py_s1_dominode_v2.shp')

    elif country == 'Nigeria':
        column = 'shapeName'
        gdf = gpd.read_file(
            '../../data/raw/shapefiles/Nigeria/geoBoundaries-NGA-ADM1-all/geoBoundaries-NGA-ADM1.shp')

    else:
        raise ValueError('Country not supported')

    if len(outcomes_of_interest) == 0:
        outcomes_of_interest = [
            'tot_asset_loss',
            'region_pml',
            'tot_exposed_asset',
            'tot_consum_loss_npv',
            'n_aff_people',
            'n_new_poor',
            'new_poverty_gap_all',
            'annual_avg_consum_loss',
            'annual_avg_consum_loss_pct',
            'n_new_poor_increase_pp',
            'n_poor_ratio',
            'r']

    # Aggregate outcomes
    if aggregation == 'mean':
        aggregated = outcomes[['region'] +
                              outcomes_of_interest].groupby('region').mean()
    elif aggregation == 'median':
        aggregated = outcomes[['region'] +
                              outcomes_of_interest].groupby('region').median()
    else:
        raise ValueError('Aggregation must be either mean or median')

    # Merge with the shapefile
    gdf = pd.merge(gdf, aggregated, left_on=column, right_index=True)
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
                       'region': 'Region'}, inplace=True)
    df.rename(columns={'annual_average_consumption_loss_pct': 'Annual average consumption loss (%)',
                       'n_new_poor': 'Number of new poor'},
              inplace=True)
    df['Policy ID'] = df['Policy'].cat.codes
    return df


def get_weeks_in_poverty_tab(outcomes: pd.DataFrame) -> pd.DataFrame:
    '''Get the average across scenarios number of weeks in poverty for each region.

    Args:
        outcomes (pd.DataFrame): Outcomes.

    Returns:
        pd.DataFrame: Average number of weeks in poverty for each region.
    '''
    # Specify the regions
    regions = ['Anse-La-Raye & Canaries', 'Castries', 'Choiseul',
               'Dennery', 'Gros Islet', 'Laborie', 'Micoud', 'Soufriere', 'Vieuxfort']
    regions = outcomes['region'].unique()

    # Keep track of the averages
    region_average = {}

    # Get the number of scenarios
    n_scenarios = outcomes['scenario'].unique().size

    # Get the keys
    column_name = 'years_in_poverty'
    all_keys = outcomes[column_name][0].keys()

    # Iterate through the regions
    for region in regions:
        # Subset the outcomes for a specific region
        df = outcomes[outcomes['region'] == region]

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
        region_average[region] = {
            key: sums[key] / n_scenarios for key in all_keys}

    # Convert the dictionary to a dataframe
    result = pd.DataFrame(region_average).T
    result.index.name = 'Region'
    result.columns = [i for i in range(0, len(all_keys))]
    # result.columns = [int(x) if int(x) < len(
    #     all_keys) else f'>{len(all_keys)}' for x in range(1, len(all_keys) + 1)]
    return result


def get_average_weighted_vulnerability(outcomes: pd.DataFrame, quintile: bool) -> pd.DataFrame:
    '''Get the average weighted vulnerability for each region.

    Args:
        outcomes (pd.DataFrame): Outcomes.
        quintile (bool): Whether to calculate the average weighted vulnerability by quintile or decile.

    Returns:
        pd.DataFrame: Average weighted vulnerability for each region.
    '''
    # Specify the regions
    regions = ['Anse-La-Raye & Canaries', 'Castries', 'Choiseul',
               'Dennery', 'Gros Islet', 'Laborie', 'Micoud', 'Soufriere', 'Vieuxfort']

    # Keep track of the averages
    region_average = {}

    # Get the number of scenarios
    n_scenarios = outcomes['scenario'].unique().size

    if quintile:
        column_name = 'weighted_vuln_quint'
        all_keys = [1, 2, 3, 4, 5]
    else:
        column_name = 'weighted_vuln_dec'
        all_keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Iterate through the regions
    for region in regions:
        # Subset the outcomes for a specific region
        df = outcomes[outcomes['region'] == region]

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
        region_average[region] = {
            key: sums[key] / n_scenarios for key in all_keys}

    # Convert the dictionary to a dataframe
    result = pd.DataFrame(region_average).T

    # Rename the index and columns
    result.index.name = 'region'
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


def split_policies(outcomes: pd.DataFrame) -> tuple:
    # TODO: Make the function more flexible and allow for different top ups and target groups
    # Split ASP policies
    asp = outcomes[outcomes['current_policy'].str.contains('asp')].copy()
    asp[['target_group', 'top_up']] = asp['current_policy'].str.split(
        '+', expand=True)
    asp['target_group'] = asp['target_group'].str.split(':', expand=True)[1]

    # Make top_up a categorical variable to ensure it is ordered correctly
    asp['top_up'] = pd.Categorical(asp['top_up'], categories=['0', '10', '50'])
    asp['target_group'] = pd.Categorical(asp['target_group'], categories=[
                                         'poor_near_poor2.0', 'all'])

    # Append None policy to ASP
    none = outcomes[outcomes['current_policy'] == 'none'].copy()
    none['target_group'] = 'none'
    none['top_up'] = 'none'
    none['current_policy'] = 'none'
    asp = pd.concat([asp, none])

    # Split retrofit policies
    retrofit = outcomes[outcomes['current_policy'].str.contains(
        'retrofit')].copy()
    retrofit[['target_group', 'houses_pct']
             ] = retrofit['current_policy'].str.split('+', expand=True)
    retrofit['target_group'] = retrofit['target_group'].str.split(':', expand=True)[
        1]
    retrofit = pd.concat([retrofit, none])
    return asp, retrofit
