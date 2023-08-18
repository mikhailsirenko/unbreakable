"""This module is used to calculate the recovery rate, consumption and well-being losses of affected households.
The calculation are done on the household level which we later aggregate to the district level."""

import numpy as np
import pandas as pd
import pickle


def calculate_recovery_rate(households: pd.DataFrame, consumption_utility: float, discount_rate: float, lambda_increment: float, years_to_recover: int) -> pd.DataFrame:
    '''Calculates the recovery rate for each affected household.

    Recovery rate is a function of the household vulnerability (v), 
    consumption utility, discount rate, median productivity and years to recover.    

    Args:
        households (pd.DataFrame): Households data frame.
        consumption_utility (float): Consumption utility.
        discount_rate (float): Discount rate.
        lambda_increment (float): Lambda increment for the integration.
        years_to_recover (int): Number of years to recover.

    Returns:
        pd.DataFrame: Households data frame with the recovery rate for each affected household.
    '''

    # Get the median productivity. It is the same for all households in a district.
    median_productivity = households['median_productivity'].values[0]

    # Subset households that are affected by the disaster
    affected_households = households[households['is_affected'] == True].copy()

    # Set effective capital stock to zero for renters
    affected_households.loc[affected_households['own_rent']
                            == 'rent', 'keff'] = 0

    # Prepare a data frame to store integration results
    data = {'aeexp': -1, 'aeexp_house': -1, 'keff': -1, 'v': -
            1, 'aesav': -1, 'solution': None, 'bankrupt': None}
    results = pd.DataFrame(data, index=[0])
    index = ['aeexp', 'aeexp_house', 'keff', 'v', 'aesav']
    results = results.reset_index(drop=True).set_index(index)

    # Calculate the recovery rate for each affected household
    affected_households['recovery_rate'] = affected_households['v'].apply(lambda x: integrate_and_find_recovery_rate(
        x, results, consumption_utility, discount_rate, median_productivity, lambda_increment, years_to_recover))

    # Set recovery rate to zero for unaffected households
    households['recovery_rate'] = 0

    # Update the recovery rate in the households data frame for affected households
    households.loc[households['is_affected'] == True,
                   'recovery_rate'] = affected_households['recovery_rate']

    return households


def integrate_and_find_recovery_rate(v: float, results: pd.DataFrame, consumption_utility: float, discount_rate: float, median_productivity: float, lambda_increment: float, years_to_recover: int) -> float:
    '''Find recovery rate (lambda) given the value of `v` (household vulnerability).

    Args:
        v (float): Household vulnerability.
        results (pd.DataFrame): Data frame to store integration results.
        consumption_utility (float): Consumption utility.
        discount_rate (float): Discount rate.
        median_productivity (float): Median productivity.
        lambda_increment (float): Lambda increment for the integration.
        years_to_recover (int): Number of years to recover.

    Returns:
        float: Recovery rate (lambda).
    '''
    rounded_v = round(v, 3)
    rounded_median_productivity = round(median_productivity, 3)

    try:
        # Look for the existing solution
        solution = results.loc[(
            0, 0, 0, rounded_v, rounded_median_productivity), 'solution']
        return solution

    except KeyError:
        # No existing solution found, so we need to optimize
        tot_weeks = 52 * years_to_recover
        dt = years_to_recover / tot_weeks

        _lambda = 0
        last_dwdlambda = 0

        while True:
            dwdlambda = 0
            for _t in np.linspace(0, years_to_recover, tot_weeks):
                factor = median_productivity + _lambda
                part1 = median_productivity - \
                    factor * v * np.e**(-_lambda * _t)
                part1 = part1**(-consumption_utility)
                part2 = _t * factor - 1
                part3 = np.e**(-_t * (discount_rate + _lambda))
                dwdlambda += part1 * part2 * part3 * dt

            if (last_dwdlambda < 0 and dwdlambda > 0) or (last_dwdlambda > 0 and dwdlambda < 0) or _lambda > 10:
                results.loc[(0, 0, 0, rounded_v, rounded_median_productivity), [
                    'solution', 'bankrupt']] = [_lambda, False]
                results = results.sort_index()
                return _lambda

            last_dwdlambda = dwdlambda
            _lambda += lambda_increment


def calculate_wellbeing(households: pd.DataFrame, consumption_utility: float, discount_rate: float, income_and_expenditure_growth: float, years_to_recover: int, add_income_loss: bool, cash_transfer: dict = {}) -> pd.DataFrame:
    '''Calculates consumption loss and well-being for each affected household.

    Args:
        households (pd.DataFrame): Households data frame.
        consumption_utility (float): Consumption utility.
        discount_rate (float): Discount rate.
        income_and_expenditure_growth (float): Income and expenditure growth.
        years_to_recover (int): Number of years to recover.
        add_income_loss (bool): Whether to add income loss or not.
        cash_transfer (dict, optional): Cash transfer. Defaults to {}, where key is the week number and value is the amount.

    Returns:
        pd.DataFrame: Households data frame with consumption loss and well-being for each affected household.
    '''

    # Get the median productivity and poverty line. They are the same for all households in a district.
    median_productivity = households['median_productivity'].values[0]
    poverty_line_adjusted = households['poverty_line_adjusted'].values[0]

    # Add new columns
    columns = ['consumption_loss',
               'consumption_loss_NPV',
               'net_consumption_loss',
               'net_consumption_loss_NPV',
               'c_t',
               'c_t_unaffected',
               'w_final',
               'weeks_pov',
               'w_final2']

    # Set values to zero
    households[columns] = 0

    # Subset households that are affected by the disaster
    affected_households = households[households['is_affected'] == True].copy()

    # Define the number of weeks given the number of years
    tot_weeks = 52 * years_to_recover
    dt = years_to_recover / tot_weeks

    # * Store consumption recovery in a dict for verification and debugging purposes
    # consumption_recovery = {}

    # We need to adjust the cash transfer to the timestep of the integration
    if cash_transfer != {}:  # If there is a cash transfer
        cash_transfer_transformed = {np.linspace(0, years_to_recover, tot_weeks)[
            t]: cash_transfer[t] for t in list(cash_transfer.keys())}
    else:
        cash_transfer_transformed = {}

    # Integrate consumption loss and well-being
    for _t in np.linspace(0, years_to_recover, tot_weeks):
        # TODO: Add an extra condition about to whom the transfer is given
        # A "dynamic policy"
        if _t in list(cash_transfer_transformed.keys()):
            affected_households['aesav'] += cash_transfer_transformed[_t]

        gfac = (1 + income_and_expenditure_growth)**_t

        expenditure_growth = gfac * affected_households['aeexp']

        exponential_multiplier = np.e**(
            -affected_households['recovery_rate']*_t)

        savings = gfac * \
            affected_households['aesav'] * affected_households['recovery_rate']

        asset_loss = gfac * \
            affected_households[['v', 'keff', 'recovery_rate']].prod(axis=1)

        # Initial implementation
        # asset_damage = gfac * \
        #     affected_households['v'] * \
        #     (affected_households['aeexp_house'] + \
        #         affected_households[['keff', 'recovery_rate']].prod(axis=1))

        income_loss = gfac * (1 - affected_households['delta_tax_safety']) * \
            median_productivity * \
            affected_households['keff'] * affected_households['v']

        if add_income_loss == False:
            affected_households['c_t'] = (expenditure_growth +
                                          exponential_multiplier * (savings - asset_loss))
        else:
            affected_households['c_t'] = (expenditure_growth +
                                          exponential_multiplier * (savings - asset_loss - income_loss))

        affected_households['c_t_unaffected'] = expenditure_growth

        # Check if any of the households has a negative consumption
        # It may happen if a household have very low savings or expenditure
        if (affected_households['c_t'] < 0).any():
            # If so, then set the consumption to 0
            affected_households.loc[affected_households['c_t'] < 0, 'c_t'] = 0
            # * Previously it was set to 1
            # affected_households.loc[affected_households['c_t'] < 1, 'c_t'] = 1

        # Consumption after the disaster should be lower than or equal to consumption before the disaster
        if (affected_households['c_t'] > affected_households['c_t_unaffected']).any():
            # In some cases there is a very small difference between the two
            # E.g. 2408.7431 vs 2408.0711
            affected_households.loc[affected_households['c_t'] > affected_households['c_t_unaffected'],
                                    'c_t'] = affected_households.loc[affected_households['c_t'] > affected_households['c_t_unaffected'], 'c_t_unaffected']

        # Total consumption loss
        affected_households['consumption_loss'] += dt * \
            (affected_households['c_t_unaffected'] -
                affected_households['c_t'])

        affected_households['consumption_loss_NPV'] += dt * \
            (affected_households['c_t_unaffected'] -
                affected_households['c_t'])*np.e**(-discount_rate*_t)

        # Net consumption loss
        affected_households['net_consumption_loss'] += dt * \
            np.e**(-affected_households['recovery_rate']*_t) * \
            affected_households['v']*gfac * \
            affected_households['aeexp_house']

        affected_households['net_consumption_loss_NPV'] += dt * \
            np.e**(-affected_households['recovery_rate']*_t) * affected_households['v']*gfac * \
            affected_households['aeexp_house'] * \
            np.e**(-discount_rate*_t)

        # Increase the number of weeks in poverty
        affected_households.loc[affected_households['c_t']
                                < poverty_line_adjusted, 'weeks_pov'] += 1

        # Integrate wellbeing
        # !!!: Fix wellbeing integration, currently it is nan
        affected_households['w_final'] += dt * (affected_households['c_t'])**(1 - consumption_utility) * \
            np.e**(-discount_rate * _t) / (1 - consumption_utility)

        affected_households['w_final2'] += affected_households['c_t_unaffected']**(1-consumption_utility)/(1-consumption_utility)*dt*(
            (1-((affected_households['c_t_unaffected'] - affected_households['c_t'])/affected_households['c_t_unaffected'])*np.e**(-affected_households['recovery_rate']*_t))**(1-consumption_utility)-1)*np.e**(-discount_rate*_t)

        # Use to examine individual consumption recovery
        # Save consumption recovery value at the time _t
        # consumption_recovery[_t] = affected_households['c_t']

    # Save consumption recovery as pickle file
    # with open('consumption_recovery.pickle', 'wb') as handle:
    #     pickle.dump(consumption_recovery, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the content of the columns for affected_households into households data frame
    households.loc[affected_households.index,
                   columns] = affected_households[columns]

    return households
