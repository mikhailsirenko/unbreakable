import numpy as np
import pandas as pd
import pickle


def calculate_recovery_rate(households: pd.DataFrame, consumption_utility: float, discount_rate: float, optimization_timestep: float, years_to_recover: int) -> pd.DataFrame:
    '''Calculates the recovery rate for each affected household.'''

    # Get the median productivity. It is the same for all households in a district.
    median_productivity = households['median_productivity'].values[0]

    # Subset households that are affected by the disaster
    affected_households = households[households['is_affected'] == True].copy()

    # Set effective capital stock to zero for renters
    affected_households.loc[affected_households['own_rent']
                            == 'rent', 'keff'] = 0

    # Prepare a data frame to store optimization results
    data = {'aeexp': -1, 'aeexp_house': -1, 'keff': -1, 'v': -
            1, 'aesav': -1, 'solution': None, 'bankrupt': None}
    optimization_results = pd.DataFrame(data, index=[0])
    index = ['aeexp', 'aeexp_house', 'keff', 'v', 'aesav']
    optimization_results = optimization_results.reset_index(
        drop=True).set_index(index)

    # Calculate the recovery rate for each affected household
    affected_households['recovery_rate'] = affected_households['v'].apply(
        lambda x: optimize_recovery_rate(x, optimization_results, consumption_utility, discount_rate, median_productivity, optimization_timestep, years_to_recover))

    # Set recovery rate to zero for unaffected households
    households['recovery_rate'] = 0

    # Update the recovery rate in the households data frame for affected households
    households.loc[households['is_affected'] == True,
                   'recovery_rate'] = affected_households['recovery_rate']

    return households


def optimize_recovery_rate(x, optimization_results: pd.DataFrame, consumption_utility: float, discount_rate: float, median_productivity: float, optimization_timestep: float, years_to_recover: int) -> float:
    try:
        # Look for the existing solution
        solution = optimization_results.loc[(
            0, 0, 0, round(x, 3), round(median_productivity, 3)), 'solution']
        return solution
    except:
        # No existing solution found, so we need to optimize
        t_max_linspace = years_to_recover  # years
        nsteps_linspace = 52 * t_max_linspace  # total weeks
        dt = t_max_linspace / nsteps_linspace

        _lambda = 0
        opt_step = optimization_timestep
        last_dwdlambda = 0

        while True:

            dwdlambda = 0

            for _t in np.linspace(0, t_max_linspace, nsteps_linspace):

                part1 = median_productivity - \
                    (median_productivity+_lambda)*x*np.e**(-_lambda*_t)
                part1 = part1**(-consumption_utility)

                part2 = _t * (median_productivity+_lambda) - 1

                part3 = np.e**(-_t*(discount_rate+_lambda))

                dwdlambda += part1 * part2 * part3 * dt

            # !: All these do the same
            if last_dwdlambda < 0 and dwdlambda > 0:
                optimization_results.loc[(0, 0, 0, round(x, 3), round(median_productivity, 3)), [
                    'solution', 'bankrupt']] = [_lambda, False]
                optimization_results = optimization_results.sort_index()
                return _lambda

            elif last_dwdlambda > 0 and dwdlambda < 0:
                optimization_results.loc[(0, 0, 0, round(x, 3), round(median_productivity, 3)), [
                    'solution', 'bankrupt']] = [_lambda, False]
                optimization_results = optimization_results.sort_index()
                return _lambda

            # !: That's why assigning more than 10 years does not work, we need to change 10 to `years_to_recover`?
            elif _lambda > 10:
                optimization_results.loc[(0, 0, 0, round(x, 3), round(median_productivity, 3)), [
                    'solution', 'bankrupt']] = [_lambda, False]
                optimization_results = optimization_results.sort_index()
                return _lambda

            else:
                last_dwdlambda = dwdlambda
            _lambda += opt_step


def integrate_wellbeing(households: pd.DataFrame, consumption_utility: float, discount_rate: float, income_and_expenditure_growth: float, years_to_recover: int, add_income_loss: bool, cash_transfer: dict = {}) -> pd.DataFrame:
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
               'w_final2',
               'asset_loss_manual']

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

    # Calculate consumption recovery-related variables on the household level
    # For debugging
    household_1 = {}
    household_2 = {}
    for _t in np.linspace(0, years_to_recover, tot_weeks):
        # Add an extra condition about to whom the transfer is given
        # # A "dynamic policy"
        # if _t in list(cash_transfer_transformed.keys()):
        #     affected_households['aesav'] += cash_transfer_transformed[_t]

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

        # Total (integrated) consumption loss
        affected_households['consumption_loss'] += dt * \
            (affected_households['c_t_unaffected'] -
                affected_households['c_t'])

        affected_households['consumption_loss_NPV'] += dt * \
            (affected_households['c_t_unaffected'] -
                affected_households['c_t'])*np.e**(-discount_rate*_t)

        # Collect consumption recovery-related variables for debugging
        try:
            idx1 = affected_households.index[0]
            idx2 = affected_households.index[1]
            district = affected_households['district'].values[0]
            district_pml = affected_households['district_pml'].values[0]

            household_1[_t] = {'district': district,
                    'district_pml': district_pml,
                    'expend_growth': expenditure_growth[idx1],
                    'exp_mult': exponential_multiplier[idx1],
                    'sav': savings[idx1],
                    'asset_loss': asset_loss[idx1],
                    'inc_loss': income_loss[idx1],
                    'c_t': affected_households['c_t'][idx1],
                    'c_t_unaffected': affected_households['c_t_unaffected'][idx1],
                    'cons_loss': affected_households['consumption_loss'][idx1],
                    'cons_loss_NPV': affected_households['consumption_loss_NPV'][idx1]}

            household_2[_t] = {'district': district,
                               'district_pml': district_pml,
                               'expend_growth': expenditure_growth[idx2],
                               'exp_mult': exponential_multiplier[idx2],
                               'sav': savings[idx2],
                               'asset_loss': asset_loss[idx2],
                               'inc_loss': income_loss[idx2],
                               'c_t': affected_households['c_t'][idx2],
                               'c_t_unaffected': affected_households['c_t_unaffected'][idx2],
                               'cons_loss': affected_households['consumption_loss'][idx2],
                               'cons_loss_NPV': affected_households['consumption_loss_NPV'][idx2]}

        except:
            pass

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
        affected_households['w_final'] += dt*(affected_households['c_t'])**(1-consumption_utility) * \
            np.e**(-discount_rate*_t)/(1-consumption_utility)

        affected_households['w_final2'] += affected_households['c_t_unaffected']**(1-consumption_utility)/(1-consumption_utility)*dt*(
            (1-((affected_households['c_t_unaffected'] - affected_households['c_t'])/affected_households['c_t_unaffected'])*np.e**(-affected_households['recovery_rate']*_t))**(1-consumption_utility)-1)*np.e**(-discount_rate*_t)

        # Use to examine individual consumption recovery
        # Save consumption recovery value at the time _t
        # consumption_recovery[_t] = affected_households['c_t']

    # Save consumption recovery as pickle file
    # with open('consumption_recovery.pickle', 'wb') as handle:
    #     pickle.dump(consumption_recovery, handle, protocol=pickle.HIGHEST_PROTOCOL)

    affected_households['asset_loss_manual'] = asset_loss

    total_consumption_loss = (affected_households['consumption_loss_NPV'] * affected_households['popwgt']).sum()
    total_asset_loss = (affected_households['asset_loss_manual'] * affected_households['popwgt']).sum()

    r = total_asset_loss / total_consumption_loss

    if r > 4:
        try:
            household_1 = pd.DataFrame(household_1).T
            household_1.to_csv('../debugging/household_1_r4.csv', index=False)
            household_2 = pd.DataFrame(household_2).T
            household_2.to_csv('../debugging/household_2_r4.csv', index=False)
        except:
            pass
    elif r < 1:
        try:
            household_1 = pd.DataFrame(household_1).T
            household_1.to_csv('../debugging/household_1_r1.csv', index=False)
            household_2 = pd.DataFrame(household_2).T
            household_2.to_csv('../debugging/household_2_r1.csv', index=False)
        except:
            pass
    else:
        pass

    # Save the content of the columns for affected_households into households data frame
    households.loc[affected_households.index, columns] = affected_households[columns]

    return households
