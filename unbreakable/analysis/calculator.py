import pandas as pd
import numpy as np
import yaml


def calculate_outcomes(households: pd.DataFrame, tot_exposed_asset: float, expected_loss_frac: float, region_pml: float, years_to_recover: int, welfare: float) -> dict:
    outcomes = {}

    tot_pop = households['wgt'].sum()
    pov_line_adjust = households['poverty_line_adjusted'].values[0]
    aff_households = households[households['is_affected'] == True]

    outcomes['return_period'] = households['return_period'].iloc[0]
    outcomes['tot_pop'] = tot_pop
    outcomes['tot_households'] = households.shape[0]
    outcomes['n_aff_people'] = aff_households['wgt'].sum()
    outcomes['n_aff_households'] = aff_households.shape[0]
    try:
        outcomes['n_retrofitted'] = households[households['retrofitted']
                                               == True]['wgt'].sum()
        outcomes['n_retrofitted_hh'] = households[households['retrofitted']
                                                  == True].shape[0]
        outcomes['n_aff_retrofitted'] = aff_households[aff_households['retrofitted']
                                                       == True]['wgt'].sum()
        outcomes['n_aff_retrofitted_hh'] = aff_households[aff_households['retrofitted'] == True].shape[0]
    except:
        outcomes['n_retrofitted_ppl'] = 0
        outcomes['n_retrofitted_hh'] = 0
        outcomes['n_aff_retrofitted_ppl'] = 0
        outcomes['n_aff_retrofitted_hh'] = 0

    outcomes['tot_asset_surv'] = households['tot_asset_surv'].iloc[0]
    outcomes['tot_exposed_asset'] = tot_exposed_asset
    outcomes['tot_asset_loss'] = aff_households[[
        'keff', 'v', 'wgt']].prod(axis=1).sum()
    outcomes['expected_loss_frac'] = expected_loss_frac
    outcomes['region_pml'] = region_pml
    outcomes['annual_avg_consum'] = weighted_mean(households, 'exp', 'wgt')
    outcomes['pov_line_adjust'] = households['poverty_line_adjusted'].iloc[0]
    outcomes['mean_recovery_rate'] = aff_households['recovery_rate'].mean()
    outcomes['tot_consum_loss_npv'] = weighted_sum(
        aff_households, 'consumption_loss_npv', 'wgt')

    # Get the model outcomes
    n_poor_initial, n_new_poor, n_poor_affected, poor_initial, new_poor = find_poor(
        households, pov_line_adjust, years_to_recover)

    years_in_poverty = get_people_by_years_in_poverty(new_poor)

    initial_poverty_gap, new_poverty_gap_initial, new_poverty_gap_all = calculate_poverty_gap(
        poor_initial, new_poor, pov_line_adjust, years_to_recover)

    annual_average_consumption_loss, annual_average_consumption_loss_pct = calculate_average_annual_consumption_loss(
        aff_households, years_to_recover)

    tot_consum_equiv_loss = - aff_households['wellbeing'].div(welfare).multiply(
        aff_households['wgt']).sum()

    r = calculate_resilience(aff_households, tot_consum_equiv_loss)

    outcomes['n_poor_initial'] = n_poor_initial
    outcomes['n_poor_affected'] = n_poor_affected
    outcomes['n_new_poor'] = n_new_poor
    outcomes['initial_poverty_gap'] = initial_poverty_gap
    outcomes['new_poverty_gap_initial'] = new_poverty_gap_initial
    outcomes['new_poverty_gap_all'] = new_poverty_gap_all
    outcomes['annual_avg_consum_loss'] = annual_average_consumption_loss
    outcomes['annual_avg_consum_loss_pct'] = annual_average_consumption_loss_pct
    outcomes['r'] = r
    outcomes['years_in_poverty'] = years_in_poverty

    # Save outcome names in a yaml file to pick up in preprocessing
    with open('analysis/outcomes.yaml', 'w') as f:
        yaml.dump(list(outcomes.keys()), f)

    return outcomes


def find_poor(households: pd.DataFrame, pov_line: float, years_to_recover: int) -> tuple:
    '''Get the poor at the beginning of the simulation and the poor at the end of the simulation

    Args:
        households (pd.DataFrame): Households.
        pov_line (float): Poverty line.
        years_to_recover (int): Number of years cut-off parameter when calculating consumption loss.

    Returns:
        tuple: Number of poor at the beginning of the simulation, number of new poor at the end of the simulation, and the new poor dataframe
    '''
    # First, find the poor at the beginning of the simulation
    poor_initial = households[households['is_poor'] == True]
    n_poor_initial = round(poor_initial['wgt'].sum())
    n_poor_affected = round(
        poor_initial[poor_initial['is_affected'] == True]['wgt'].sum())

    # Second, find the new poor at the end of the simulation (`years_to_recover`)
    not_poor = households[households['is_poor'] == False]
    not_poor_affected = not_poor[not_poor['is_affected'] == True]
    x = not_poor_affected['exp'] - \
        not_poor_affected['consumption_loss_npv'] / years_to_recover
    new_poor = not_poor_affected.loc[x < pov_line, :]
    new_poor = new_poor.assign(is_poor=True)
    n_new_poor = round(new_poor['wgt'].sum())

    return n_poor_initial, n_new_poor, n_poor_affected, poor_initial, new_poor


def get_people_by_years_in_poverty(affected_households: pd.DataFrame) -> dict:
    '''Get the number of people in poverty for each year in poverty.

    Args:
        affected_households (pd.DataFrame): Affected households

    Returns:
        dict: Number of people in poverty for each year in poverty
    '''
    affected_households = affected_households.assign(
        years_in_poverty=affected_households['weeks_in_poverty'] // 52)
    d = {}
    longest_years_in_poverty = 10
    for i in range(longest_years_in_poverty + 1):
        d[i] = round(affected_households[affected_households['years_in_poverty'] == i]
                     ['wgt'].sum())
    return d


def calculate_poverty_gap(poor_initial: pd.DataFrame, new_poor: pd.DataFrame, poverty_line: float, years_to_recover: int) -> tuple:
    '''Calculate the poverty gap at the beginning and at the end of the simulation.

    Args:
        poor_initial (pd.DataFrame): Poor at the beginning of the simulation
        new_poor (pd.DataFrame): New poor at the end of the simulation
        poverty_line (float): Poverty line
        years_to_recover (int): Number of years cut-off parameter when calculating consumption loss. Default is 10.

    Returns:
        tuple: Poverty gap at the beginning and at the end of the simulation

    Raises:
        Exception: If the index is duplicated
        Exception: If the poverty gap is greater than 1
    '''
    average_expenditure_poor_initial = (
        poor_initial['exp'] * poor_initial['wgt']).sum() / poor_initial['wgt'].sum()

    # assert poverty_line > average_expenditure_poor_initial, 'Poverty line cannot be less than average expenditure of the poor'

    initial_poverty_gap = (
        poverty_line - average_expenditure_poor_initial) / poverty_line

    # Combine the poor at the beginning of the simulation and the new poor at the end of the simulation
    all_poor = pd.concat([poor_initial, new_poor])

    # Expenditure of both were affected by the disaster
    all_poor = all_poor.assign(
        exp=all_poor['exp'] - all_poor['consumption_loss_npv'] / years_to_recover)

    # Now, get the average expenditure of the poor at the end of the simulation
    average_expenditure_poor_all = (
        all_poor['exp'] * all_poor['wgt']).sum() / all_poor['wgt'].sum()

    # Calculate the poverty gap at the end of the simulation
    new_poverty_gap_all = (
        poverty_line - average_expenditure_poor_all) / poverty_line

    # However, we also want to know the poverty gap for old poor
    poor_initial = poor_initial.assign(
        exp=poor_initial['exp'] - poor_initial['consumption_loss_npv'] / years_to_recover)

    average_expenditure_poor_initial = (
        poor_initial['exp'] * poor_initial['wgt']).sum() / poor_initial['wgt'].sum()

    new_poverty_gap_initial = (
        poverty_line - average_expenditure_poor_initial) / poverty_line

    # Poverty gap cannot be greater than 1
    if initial_poverty_gap > 1 or new_poverty_gap_initial > 1 or new_poverty_gap_all > 1:
        raise Exception('Poverty gap cannot be greater than 1')

    return initial_poverty_gap, new_poverty_gap_initial, new_poverty_gap_all


def calculate_average_annual_consumption_loss(affected_households: pd.DataFrame, years_to_recover: int) -> tuple:
    '''Get the average annual consumption loss and the average annual consumption loss as a percentage of average annual consumption.

    Args:
        affected_households (pd.DataFrame): Affected households.
        years_to_recover (int): Number of years cut-off parameter when calculating consumption loss. Default is 10. 

    Returns:
        tuple: Average annual consumption loss and average annual consumption loss as a percentage of average annual consumption

    Raises:
        Exception: If the average annual consumption loss is greater than 1
    '''

    if len(affected_households) == 0:
        return np.nan, np.nan

    # Annual consumption loss
    annual_consumption_loss = (
        affected_households['consumption_loss_npv'].div(years_to_recover).multiply(affected_households['wgt'])).sum()

    # Weighted average
    annual_average_consumption_loss = annual_consumption_loss / \
        affected_households['wgt'].sum()

    annual_average_consumption_loss_pct = (affected_households['consumption_loss_npv']
                                           .div(years_to_recover)
                                           .div(affected_households['exp'])
                                           .multiply(affected_households['wgt']).sum())\
        / affected_households['wgt'].sum()

    if annual_average_consumption_loss_pct > 1:
        raise Exception(
            'Annual average consumption loss is greater than 1')

    return annual_average_consumption_loss, annual_average_consumption_loss_pct


def calculate_resilience(affected_households: pd.DataFrame, tot_consum_equiv_loss: float) -> float:
    '''Calculate socio-economic resilience of affected households.

    Socio-economic resilience is a ratio of asset loss to consumption loss.

    Args:
        affected_households (pd.DataFrame): Affected households.
        tot_consum_equiv_loss (float): Total consumption equivalent loss.

    Returns:
        float: Socio-economic resilience
    '''
    # TODO: Test resilience values, it should not go above, e.g., 10
    # total_consumption_loss = (
    #     affected_households[['consumption_loss_NPV', 'wgt']].prod(axis=1)).sum()

    total_asset_damage = (
        affected_households[['keff', 'v', 'wgt']].prod(axis=1)).sum()

    # if total_consumption_loss == 0:
    #     r = np.nan

    # else:
    # r = total_asset_damage / total_consumption_loss
    r = total_asset_damage / tot_consum_equiv_loss

    return r


def weighted_sum(df: pd.DataFrame, col: str, wgt: str) -> float:
    '''Calculate the weighted sum of a column in a dataframe.

    Args:
        df (pd.DataFrame): Dataframe.
        col (str): Column name.
        wgt (str): Weight column name.

    Returns:
        float: Weighted sum of a column in a dataframe.
    '''
    return (df[col] * df[wgt]).sum()


def weighted_mean(df: pd.DataFrame, col: str, wgt: str) -> float:
    '''Calculate the weighted mean of a column in a dataframe.

    Args:
        df (pd.DataFrame): Dataframe.
        col (str): Column name.
        wgt (str): Weight column name.

    Returns:
        float: Weighted mean of a column in a dataframe.
    '''
    return weighted_sum(df, col, wgt) / df[wgt].sum()
