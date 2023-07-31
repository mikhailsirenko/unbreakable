# Specify which outcomes to store in each run of the simulation model

import pandas as pd
import numpy as np


def prepare_outcomes(households: pd.DataFrame, affected_households: pd.DataFrame) -> pd.DataFrame:
    '''Add columns/outcomes of interest from `affected households` to the `households` dataframe.'''

    outcomes_of_interest: list = [
        'consumption_loss',
        'consumption_loss_NPV',
        'c_t',
        'c_t_unaffected',
        'recovery_rate',
        'weeks_pov']
    columns = ['hhid'] + outcomes_of_interest
    households = pd.merge(
        households, affected_households[columns], on='hhid', how='left')
    return households


def get_outcomes(households, event_damage, total_asset_stock, expected_loss_fraction, average_productivity, x_max) -> dict:
    '''Calculate outcomes of interest from the simulation model.

    Args:
        households (pd.DataFrame): Households data frame.
        event_damage (float): Event damage.
        total_asset_stock (float): Total asset stock.
        expected_loss_fraction (float): Expected loss fraction.
        average_productivity (float): Average productivity.
        x_max (float): Number of years to consider for calculations (same as for optimization algorithm).

    Returns:
        dict: Outcomes of interest.
    '''
    # Save some outcomes for verification
    total_population = households['popwgt'].sum()
    total_asset_in_survey = households['total_asset_in_survey'].iloc[0]

    # Actual outcomes of interest
    affected_households = households[households['is_affected'] == True]
    total_asset_loss = affected_households[[
        'keff', 'v', 'popwgt']].prod(axis=1).sum()
    total_consumption_loss = affected_households[[
        'consumption_loss_NPV', 'popwgt']].prod(axis=1).sum()
    n_affected_people = affected_households['popwgt'].sum()
    annual_average_consumption = (
        households['aeexp'] * households['popwgt']).sum() / households['popwgt'].sum()
    recovery_rate = households['recovery_rate'].mean()

    # * Poverty line is different across replications
    poverty_line_adjusted = households['poverty_line_adjusted'].values[0]

    # Get PML, its the same across replications and stored in households
    pml = households['pml'].iloc[0]

    # Run statistics
    no_affected_households = 0
    zero_consumption_loss = 0

    # * Some runs give no affected households and we will skip these
    if len(affected_households) == 0:
        no_affected_households += 1
        pass

    # * Sometimes households are affected but they have no consumption loss
    if affected_households['consumption_loss_NPV'].sum() == 0:
        zero_consumption_loss += 1
        pass

    # Calculate outcomes of interest
    n_poor_initial, n_new_poor, n_poor_affected, poor_initial, new_poor = find_poor(
        households, poverty_line_adjusted, x_max)

    years_in_poverty = get_people_by_years_in_poverty(new_poor)

    initial_poverty_gap, new_poverty_gap = calculate_poverty_gap(
        poor_initial, new_poor, poverty_line_adjusted, x_max)

    annual_average_consumption_loss, annual_average_consumption_loss_pct = calculate_average_annual_consumption_loss(
        affected_households, x_max)

    r = calculate_resilience(
        affected_households, pml)

    return {
        'total_population': total_population,
        'total_asset_loss': total_asset_loss,
        'total_consumption_loss': total_consumption_loss,
        'event_damage': event_damage,
        'total_asset_stock': total_asset_stock,
        'average_productivity': average_productivity,
        'total_asset_in_survey': total_asset_in_survey,
        'expected_loss_fraction': expected_loss_fraction,
        'n_affected_people': n_affected_people,
        'annual_average_consumption': annual_average_consumption,
        'poverty_line_adjusted': poverty_line_adjusted,
        'pml': pml,
        'n_poor_initial': n_poor_initial,
        'n_poor_affected': n_poor_affected,
        'n_new_poor': n_new_poor,
        'initial_poverty_gap': initial_poverty_gap,
        'new_poverty_gap': new_poverty_gap,
        'annual_average_consumption_loss': annual_average_consumption_loss,
        'annual_average_consumption_loss_pct': annual_average_consumption_loss_pct,
        'r': r,
        'recovery_rate': recovery_rate,
        'years_in_poverty': years_in_poverty
        # 'n_resilience_more_than_1' : n_resilience_more_than_1
    }


def find_poor(households: pd.DataFrame, poverty_line: float, x_max: int) -> tuple:
    '''Get the poor at the beginning of the simulation and the poor at the end of the simulation

    Args:
        households (pd.DataFrame): Household dataframe
        poverty_line (float): Poverty line

    Returns:
        tuple: Number of poor at the beginning of the simulation, number of new poor at the end of the simulation, and the new poor dataframe
    '''
    # First, find the poor at the beginning of the simulation
    poor_initial = households[households['is_poor'] == True]
    n_poor_initial = round(poor_initial['popwgt'].sum())
    n_poor_affected = round(
        poor_initial[poor_initial['is_affected'] == True]['popwgt'].sum())

    # Second, find the new poor at the end of the simulation (`x_max``)
    not_poor = households[households['is_poor'] == False]
    not_poor_affected = not_poor[not_poor['is_affected'] == True]
    x = not_poor_affected['aeexp'] - \
        not_poor_affected['consumption_loss_NPV'] / x_max
    new_poor = not_poor_affected.loc[x < poverty_line, :]
    n_new_poor = round(new_poor['popwgt'].sum())

    return n_poor_initial, n_new_poor, n_poor_affected, poor_initial, new_poor


def get_people_by_years_in_poverty(new_poor: pd.DataFrame) -> dict:
    '''Get the number of people in poverty for each year in poverty.

    Args:
        new_poor (pd.DataFrame): New poor dataframe

    Returns:
        dict: Number of people in poverty for each year in poverty
    '''
    new_poor = new_poor.assign(
        years_in_poverty=new_poor['weeks_pov'] // 52)
    d = {}
    # !: This cannot be higher > x_max
    longest_years_in_poverty = 10
    for i in range(longest_years_in_poverty):
        d[i] = round(new_poor[new_poor['years_in_poverty'] == i]
                     ['popwgt'].sum())
    d[longest_years_in_poverty] = round(
        new_poor[new_poor['years_in_poverty'] >= longest_years_in_poverty]['popwgt'].sum())

    return d


def calculate_poverty_gap(poor_initial: pd.DataFrame, new_poor: pd.DataFrame, poverty_line: float, x_max: int) -> tuple:
    '''Calculate the poverty gap at the beginning and at the end of the simulation.

    Args:
        poor_initial (pd.DataFrame): Poor at the beginning of the simulation
        new_poor (pd.DataFrame): New poor at the end of the simulation
        poverty_line (float): Poverty line
        x_max (int): Number of years of the optimization algorithm

    Returns:
        tuple: Poverty gap at the beginning and at the end of the simulation

    Raises:
        Exception: If the index is duplicated
        Exception: If the poverty gap is greater than 1
    '''
    # First, get the average expenditure of the poor at the beginning of the simulation
    average_expenditure_poor_initial = (
        poor_initial['aeexp'] * poor_initial['popwgt']).sum() / poor_initial['popwgt'].sum()
    initial_poverty_gap = (
        poverty_line - average_expenditure_poor_initial) / poverty_line

    new_poor['aeexp'] = new_poor['aeexp'] - \
        new_poor['consumption_loss_NPV'] / x_max

    all_poor = pd.concat([poor_initial, new_poor])

    # Now, get the average expenditure of the poor at the end of the simulation
    average_expenditure_poor_new = (
        all_poor['aeexp'] * all_poor['popwgt']).sum() / all_poor['popwgt'].sum()
    new_poverty_gap = (
        poverty_line - average_expenditure_poor_new) / poverty_line

    # Poverty gap cannot be greater than 1
    if new_poverty_gap > 1 or initial_poverty_gap > 1:
        raise Exception('Poverty gap is greater than 1')

    return initial_poverty_gap, new_poverty_gap


def calculate_average_annual_consumption_loss(affected_households: pd.DataFrame, x_max: int) -> tuple:
    '''Get the average annual consumption loss and the average annual consumption loss as a percentage of average annual consumption.

    Args:
        affected_households (pd.DataFrame): Affected households dataframe
        x_max (int): Number of years of the optimization algorithm

    Returns:
        tuple: Average annual consumption loss and average annual consumption loss as a percentage of average annual consumption

    Raises:
        Exception: If the average annual consumption loss is greater than 1
    '''

    if len(affected_households) == 0:
        return np.nan, np.nan

    # SUM(Total consumption loss / number of years * population weight)
    annual_consumption_loss = (
        affected_households['consumption_loss_NPV'].div(x_max).multiply(affected_households['popwgt'])).sum()

    # Annual consumption loss / population weight
    annual_average_consumption_loss = annual_consumption_loss / \
        affected_households['popwgt'].sum()

    annual_average_consumption = (
        affected_households['aeexp'] * affected_households['popwgt']).sum() / \
        affected_households['popwgt'].sum()

    annual_average_consumption_loss_pct = annual_average_consumption_loss / \
        annual_average_consumption

    if annual_average_consumption_loss_pct > 1:
        raise Exception(
            'Annual average consumption loss is greater than 1')

    return annual_average_consumption_loss, annual_average_consumption_loss_pct


def calculate_resilience(affected_households: pd.DataFrame, pml: float,
                         # n_resilience_more_than_1: int
                         ) -> tuple:
    '''Calculate the resilience of the affected households.

    Args:
        affected_households (pd.DataFrame): Affected households dataframe
        pml (float): Probable maximum loss

    Returns:
        tuple: Resilience and number of times resilience is greater than 1

    Raises:
        Exception: If the total consumption loss is 0
    '''
    total_consumption_loss = (
        affected_households['consumption_loss_NPV'] * affected_households['popwgt']).sum()

    if total_consumption_loss == 0:
        # raise Exception('Total consumption loss is 0')
        r = np.nan
    else:
        r = pml / total_consumption_loss

    # !: Sometimes resilience is greater than 1
    # We will set it to 1 then
    # if r > 5:
    #     r = 1
    #     # raise Exception('Resilience is greater than 1')
    #     # n_resilience_more_than_1 += 1
    #     # continue
    return r
