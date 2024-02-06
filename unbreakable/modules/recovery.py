import numpy as np
import pandas as pd
import pickle
import os


def calc_recovery_rate(households: pd.DataFrame, avg_prod: float, cons_util: float, disc_rate: float, lambda_incr: float, yrs_to_rec: int) -> pd.DataFrame:
    '''Calculate the recovery rate for each affected household.

    Recovery rate is a function of the household vulnerability (v), 
    consumption utility, discount rate, average productivity and years to recover.    

    Args:
        households (pd.DataFrame): Households data frame.
        cons_util (float): Consumption utility.
        disc_rate (float): Discount rate.
        lambda_incr (float): Lambda increment for the integration.
        yrs_to_rec (int): Number of years to recover.

    Returns:
        pd.DataFrame: Households data frame with the recovery rate for each affected household.
    '''
    # Subset households that are affected by the disaster
    affected_households = households[households['is_affected'] == True].copy()

    # Set effective capital stock to zero for renters
    affected_households.loc[affected_households['own_rent']
                            == 'rent', 'keff'] = 0

    # Prepare a data frame to store integration results
    data = {'exp': -1, 'exp_house': -1, 'keff': -1, 'v': -
            1, 'sav': -1, 'solution': None, 'bankrupt': None}

    results = pd.DataFrame(data, index=[0])
    index = ['exp', 'exp_house', 'keff', 'v', 'sav']
    results = results.reset_index(drop=True).set_index(index)

    # Calculate the recovery rate for each affected household
    affected_households['recovery_rate'] = affected_households['v'].apply(lambda x: integrate_and_find_recovery_rate(
        x, results, cons_util, disc_rate, avg_prod, lambda_incr, yrs_to_rec))

    # Set recovery rate to zero for unaffected households
    households['recovery_rate'] = 0

    # Update the recovery rate in the households data frame for affected households
    households.loc[households['is_affected'] == True,
                   'recovery_rate'] = affected_households['recovery_rate']

    return households


def integrate_and_find_recovery_rate(v: float, results: pd.DataFrame, cons_util: float, disc_rate: float, avg_prod: float, lambda_incr: float, yrs_to_rec: int) -> float:
    '''Find recovery rate (lambda) given the value of `v` (household vulnerability).

    Args:
        v (float): Household vulnerability.
        results (pd.DataFrame): Data frame to store integration results.
        cons_util (float): Consumption utility.
        disc_rate (float): Discount rate.
        avg_prod (float): Average productivity.
        lambda_incr (float): Lambda increment for the integration.
        yrs_to_rec (int): Number of years to recover.

    Returns:
        float: Recovery rate (lambda).
    '''
    rounded_v = round(v, 3)
    rounded_average_productivity = round(avg_prod, 3)

    try:
        # Look for the existing solution
        solution = results.loc[(
            0, 0, 0, rounded_v, rounded_average_productivity), 'solution']
        return solution

    except KeyError:
        # No existing solution found, so we need to optimize
        tot_weeks = 52 * yrs_to_rec
        dt = yrs_to_rec / tot_weeks

        _lambda = 0
        last_dwdlambda = 0

        while True:
            dwdlambda = 0
            for _t in np.linspace(0, yrs_to_rec, tot_weeks):
                factor = avg_prod + _lambda
                part1 = avg_prod - \
                    factor * v * np.e**(-_lambda * _t)
                part1 = part1**(-cons_util)
                part2 = _t * factor - 1
                part3 = np.e**(-_t * (disc_rate + _lambda))
                dwdlambda += part1 * part2 * part3 * dt

            if (last_dwdlambda < 0 and dwdlambda > 0) or (last_dwdlambda > 0 and dwdlambda < 0) or _lambda > 10:
                results.loc[(0, 0, 0, rounded_v, rounded_average_productivity), [
                    'solution', 'bankrupt']] = [_lambda, False]
                results = results.sort_index()
                return _lambda

            last_dwdlambda = dwdlambda
            _lambda += lambda_incr


def calc_wellbeing(households: pd.DataFrame, avg_prod: float, cons_util: float, disc_rate: float, inc_exp_growth: float, yrs_to_rec: int, add_inc_loss: bool, cash_transfer: dict = {}) -> pd.DataFrame:
    '''Calculates consumption loss and well-being for each affected household.

    Args:
        households (pd.DataFrame): Households data frame.
        cons_util (float): Consumption utility.
        disc_rate (float): Discount rate.
        inc_exp_growth (float): Income and expenditure growth.
        yrs_to_rec (int): Number of years to recover.
        add_inc_loss (bool): Whether to add income loss or not.
        cash_transfer (dict, optional): Cash transfer. Defaults to {}, where key is the week number and value is the amount.

    Returns:
        pd.DataFrame: Households data frame with consumption loss and well-being for each affected household.
    '''
    # Get the poverty line. It is the same for all households
    povline_adjusted = households['povline_adjusted'].values[0]

    # Add new columns
    columns = ['consumption_loss',
               'consumption_loss_NPV',
               'net_consumption_loss',
               'net_consumption_loss_NPV',
               'c_t',
               'c_t_unaffected',
               'weeks_pov',
               'wellbeing']

    # Set values to zero
    households[columns] = 0

    # Subset households that are affected by the disaster
    affected_households = households[households['is_affected'] == True].copy()

    # Define the number of weeks given the number of years
    tot_weeks = 52 * yrs_to_rec
    dt = yrs_to_rec / tot_weeks

    # NOTE: Store consumption recovery in a dict for verification and debugging purposes
    consumption_recovery = {}

    # We need to adjust the cash transfer to the timestep of the integration
    if cash_transfer != {}:  # If there is a cash transfer
        cash_transfer_transformed = {np.linspace(0, yrs_to_rec, tot_weeks)[
            t]: cash_transfer[t] for t in list(cash_transfer.keys())}
    else:
        cash_transfer_transformed = {}

    # Integrate consumption loss and well-being
    for _t in np.linspace(0, yrs_to_rec, tot_weeks):
        # TODO: Add an extra condition about to whom the transfer is given
        # A "dynamic policy"
        if _t in list(cash_transfer_transformed.keys()):
            affected_households['sav'] += cash_transfer_transformed[_t]

        # !: Should c_t_unaffected continue to grow every time step?
        gfac = (1 + inc_exp_growth)**_t

        expenditure_growth = gfac * affected_households['exp']

        exponential_multiplier = np.e**(
            -affected_households['recovery_rate']*_t)

        savings = gfac * \
            affected_households['sav'] * affected_households['recovery_rate']

        asset_loss = gfac * \
            affected_households[['v', 'keff', 'recovery_rate']].prod(axis=1)

        # NOTE: That was initial implementation
        # asset_damage = gfac * \
        #     affected_households['v'] * \
        #     (affected_households['aeexp_house'] + \
        #         affected_households[['keff', 'recovery_rate']].prod(axis=1))

        income_loss = gfac * (1 - affected_households['delta_tax_safety']) * \
            avg_prod * \
            affected_households['keff'] * affected_households['v']

        if add_inc_loss == False:
            affected_households['c_t'] = (expenditure_growth +
                                          exponential_multiplier * (savings - asset_loss))
        else:
            affected_households['c_t'] = (expenditure_growth +
                                          exponential_multiplier * (savings - asset_loss - income_loss))

        affected_households['c_t_unaffected'] = expenditure_growth

        # Check if any of the households has a negative consumption
        # It may happen if a household have very low savings or expenditure
        if (affected_households['c_t'] < 0).any():
            # If so, then set the consumption to 1
            # We must have 1 to to avoid -inf in the wellbeing integration
            # TODO: Experiment with 0.1 instead of 1
            # This could be relevant if we do daily consumption/integration
            affected_households.loc[affected_households['c_t'] < 0, 'c_t'] = 1

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
                affected_households['c_t'])*np.e**(-disc_rate*_t)

        # Net consumption loss
        affected_households['net_consumption_loss'] += dt * \
            np.e**(-affected_households['recovery_rate']*_t) * \
            affected_households['v']*gfac * \
            affected_households['exp_house']

        affected_households['net_consumption_loss_NPV'] += dt * \
            np.e**(-affected_households['recovery_rate']*_t) * affected_households['v']*gfac * \
            affected_households['exp_house'] * \
            np.e**(-disc_rate*_t)

        # Increase the number of weeks in poverty
        affected_households.loc[affected_households['c_t']
                                < povline_adjusted, 'weeks_pov'] += 1

        # Integrate well-being
        affected_households['wellbeing'] += affected_households['c_t_unaffected']**(1 - cons_util)\
            / (1 - cons_util) * dt\
            * ((1 - ((affected_households['c_t_unaffected'] - affected_households['c_t']) / affected_households['c_t_unaffected'])
                * np.e**(-affected_households['recovery_rate'] * _t))**(1 - cons_util) - 1)\
            * np.e**(-disc_rate * _t)

        # Use to examine individual consumption recovery
        # Save consumption recovery value at the time _t
        consumption_recovery[_t] = affected_households.loc[:, [
            'is_poor', 'recovery_rate', 'c_t_unaffected', 'c_t']]

    country = 'Nigeria'
    region = households['region'].values[0]
    random_seed = households['random_seed'].values[0]

    # Construct folder name
    folder = f'../experiments/{country}/cons_rec/{random_seed}'

    # Create a folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save consumption recovery as pickle file
    with open(folder + f'/{region}.pickle', 'wb') as handle:
        pickle.dump(consumption_recovery, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # Save the content of the columns for affected_households into households data frame
    households.loc[affected_households.index,
                   columns] = affected_households[columns]

    return households
