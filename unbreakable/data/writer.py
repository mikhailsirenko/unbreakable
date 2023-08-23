"""This module contains functions to calculate outcomes of interest.
It has the main function `get_outcomes` which uses other functions to calculate individual outcomes."""

import pandas as pd
import numpy as np


def get_outcomes(households, tot_exposed_asset, expected_loss_frac, years_to_recover) -> dict:
    '''Calculate outcomes of interest from the simulation model.

    Args:
        households (pd.DataFrame): Households.
        tot_exposed_asset (float): Total exposed asset stock.
        expected_loss_fraction (float): Expected loss fraction.
        years_to_recover (float): Number of years cut-off parameter when calculating consumption loss. Default is 10.

    Returns:
        dict: Outcomes of interest, where the key is the name of the outcome and value is the outcome.
    '''
    # Save some values for verification. There are not going to be used for the analysis.
    total_population = households['popwgt'].sum()
    average_productivity = households['average_productivity'].iloc[0]
    tot_asset_surv = households['tot_asset_surv'].iloc[0]
    annual_average_consumption = (
        households['aeexp'] * households['popwgt']).sum() / households['popwgt'].sum()

    affected_households = households[households['is_affected'] == True]
    n_affected_people = affected_households['popwgt'].sum()

    # Recovery rate must be calculated only for the affected households only
    mean_recovery_rate = affected_households['recovery_rate'].mean()
    total_asset_loss = affected_households[[
        'keff', 'v', 'popwgt']].prod(axis=1).sum()
    total_consumption_loss = affected_households[[
        'consumption_loss_NPV', 'popwgt']].prod(axis=1).sum()

    # Poverty line adjusted (see `match_assets_and_expenditures` function in `households.py`)
    # Poverty line ois the same for all households
    poverty_line_adjusted = households['poverty_line_adjusted'].values[0]

    n_poor_initial, n_new_poor, n_poor_affected, poor_initial, new_poor = find_poor(
        households, poverty_line_adjusted, years_to_recover)

    years_in_poverty = get_people_by_years_in_poverty(new_poor)

    initial_poverty_gap, new_poverty_gap_initial, new_poverty_gap_all = calculate_poverty_gap(
        poor_initial, new_poor, poverty_line_adjusted, years_to_recover)

    annual_average_consumption_loss, annual_average_consumption_loss_pct = calculate_average_annual_consumption_loss(
        affected_households, years_to_recover)

    r = calculate_resilience(affected_households)

    # PML is the same for all households in a district
    district_pml = households['district_pml'].iloc[0]

    weighted_vuln_quint = get_weighted_vuln(affected_households, quintile=True)
    weighted_vuln_dec = get_weighted_vuln(affected_households, quintile=False)

    return {
        'total_population': total_population,
        'total_asset_loss': total_asset_loss,
        'total_consumption_loss': total_consumption_loss,
        'tot_exposed_asset': tot_exposed_asset,
        'average_productivity': average_productivity,
        'tot_asset_surv': tot_asset_surv,
        'expected_loss_frac': expected_loss_frac,
        'n_affected_people': n_affected_people,
        'annual_average_consumption': annual_average_consumption,
        'poverty_line_adjusted': poverty_line_adjusted,
        'district_pml': district_pml,
        'n_poor_initial': n_poor_initial,
        'n_poor_affected': n_poor_affected,
        'n_new_poor': n_new_poor,
        'initial_poverty_gap': initial_poverty_gap,
        'new_poverty_gap_initial': new_poverty_gap_initial,
        'new_poverty_gap_all': new_poverty_gap_all,
        'annual_average_consumption_loss': annual_average_consumption_loss,
        'annual_average_consumption_loss_pct': annual_average_consumption_loss_pct,
        'r': r,
        'mean_recovery_rate': mean_recovery_rate,
        'weighted_vuln_quint': weighted_vuln_quint,
        'weighted_vuln_dec': weighted_vuln_dec,
        'years_in_poverty': years_in_poverty
    }


def find_poor(households: pd.DataFrame, poverty_line: float, years_to_recover: int) -> tuple:
    '''Get the poor at the beginning of the simulation and the poor at the end of the simulation

    Args:
        households (pd.DataFrame): Households.
        poverty_line (float): Poverty line.
        years_to_recover (int): Number of years cut-off parameter when calculating consumption loss.

    Returns:
        tuple: Number of poor at the beginning of the simulation, number of new poor at the end of the simulation, and the new poor dataframe
    '''
    # First, find the poor at the beginning of the simulation
    poor_initial = households[households['is_poor'] == True]
    n_poor_initial = round(poor_initial['popwgt'].sum())
    n_poor_affected = round(
        poor_initial[poor_initial['is_affected'] == True]['popwgt'].sum())

    # Second, find the new poor at the end of the simulation (`years_to_recover`)
    not_poor = households[households['is_poor'] == False]
    not_poor_affected = not_poor[not_poor['is_affected'] == True]
    x = not_poor_affected['aeexp'] - \
        not_poor_affected['consumption_loss_NPV'] / years_to_recover
    new_poor = not_poor_affected.loc[x < poverty_line, :]
    new_poor = new_poor.assign(is_poor=True)
    n_new_poor = round(new_poor['popwgt'].sum())

    return n_poor_initial, n_new_poor, n_poor_affected, poor_initial, new_poor


def get_people_by_years_in_poverty(affected_households: pd.DataFrame) -> dict:
    '''Get the number of people in poverty for each year in poverty.

    Args:
        affected_households (pd.DataFrame): Affected households

    Returns:
        dict: Number of people in poverty for each year in poverty
    '''
    affected_households = affected_households.assign(
        years_in_poverty=affected_households['weeks_pov'] // 52)
    d = {}
    # !: This cannot be higher > years_to_recover
    longest_years_in_poverty = 10
    for i in range(longest_years_in_poverty):
        d[i] = round(affected_households[affected_households['years_in_poverty'] == i]
                     ['popwgt'].sum())

    # d[longest_years_in_poverty] = round(
    #     affected_households[affected_households['years_in_poverty'] >= longest_years_in_poverty]['popwgt'].sum())

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
    # First we need to calculate the poverty gap at the beginning of the simulation
    # Get the weighted average expenditure
    average_expenditure_poor_initial = (
        poor_initial['aeexp'] * poor_initial['popwgt']).sum() / poor_initial['popwgt'].sum()

    # Calculate the poverty gap at the beginning of the simulation
    initial_poverty_gap = (
        poverty_line - average_expenditure_poor_initial) / poverty_line

    # Combine the poor at the beginning of the simulation and the new poor at the end of the simulation
    all_poor = pd.concat([poor_initial, new_poor])

    # Expenditure of both were affected by the disaster
    all_poor = all_poor.assign(
        aeexp=all_poor['aeexp'] - all_poor['consumption_loss_NPV'] / years_to_recover)

    # Now, get the average expenditure of the poor at the end of the simulation
    average_expenditure_poor_all = (
        all_poor['aeexp'] * all_poor['popwgt']).sum() / all_poor['popwgt'].sum()

    # Calculate the poverty gap at the end of the simulation
    new_poverty_gap_all = (
        poverty_line - average_expenditure_poor_all) / poverty_line

    # However, we also want to know the poverty gap for old poor
    poor_initial = poor_initial.assign(
        aeexp=poor_initial['aeexp'] - poor_initial['consumption_loss_NPV'] / years_to_recover)

    average_expenditure_poor_initial = (
        poor_initial['aeexp'] * poor_initial['popwgt']).sum() / poor_initial['popwgt'].sum()

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
        affected_households['consumption_loss_NPV'].div(years_to_recover).multiply(affected_households['popwgt'])).sum()

    # Weighted average
    annual_average_consumption_loss = annual_consumption_loss / \
        affected_households['popwgt'].sum()

    # Annual average consumption
    annual_average_consumption = (
        affected_households['aeexp'] * affected_households['popwgt']).sum() / \
        affected_households['popwgt'].sum()

    # Annual average consumption loss as a percentage of average annual consumption
    # annual_average_consumption_loss_pct = annual_average_consumption_loss / \
    #     annual_average_consumption

    annual_average_consumption_loss_pct = (affected_households['consumption_loss_NPV']
                                           .div(years_to_recover)
                                           .div(affected_households['aeexp'])
                                           .multiply(affected_households['popwgt']).sum())\
        / affected_households['popwgt'].sum()

    if annual_average_consumption_loss_pct > 1:
        raise Exception(
            'Annual average consumption loss is greater than 1')

    return annual_average_consumption_loss, annual_average_consumption_loss_pct


def calculate_resilience(affected_households: pd.DataFrame) -> float:
    '''Calculate socio-economic resilience of affected households.

    Socio-economic resilience is a ratio of asset loss to consumption loss.

    Args:
        affected_households (pd.DataFrame): Affected households.

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

    return r


def get_weighted_vuln(affected_households: pd.DataFrame, quintile: bool) -> dict:
    '''Calculate weighted average vulnerability of affected households by consumption quintile or decile.

    Args:
        affected_households (pd.DataFrame): Affected households.
        quintile (bool): Whether to calculate by quintile.

    Returns:
        dict: Weighted average vulnerability by consumption quintile or decile.
    '''
    df = affected_households.copy()
    if quintile:
        df['v_weighted'] = df['v'].multiply(df['popwgt'])
        v_weighted_by_q = df.groupby('quintile').sum(
            numeric_only=True)[['v_weighted']]
        pop_by_q = df.groupby('quintile').sum(numeric_only=True)[['popwgt']]
        average_v_by_q = v_weighted_by_q['v_weighted'].div(pop_by_q['popwgt'])
        return average_v_by_q.to_dict()
    else:
        df['v_weighted'] = df['v'].multiply(df['popwgt'])
        v_weighted_by_d = df.groupby('decile').sum(
            numeric_only=True)[['v_weighted']]
        pop_by_d = df.groupby('decile').sum(numeric_only=True)[['popwgt']]
        average_v_by_d = v_weighted_by_d['v_weighted'].div(pop_by_d['popwgt'])
        return average_v_by_d.to_dict()
