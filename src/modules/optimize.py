import numpy as np
import pandas as pd
import pickle


def run_optimization(affected_households: pd.DataFrame, consumption_utility: float, discount_rate: float, average_productivity: float, optimization_timestep: float, n_years: int) -> pd.DataFrame:
    '''This function calculates the recovery rate for each affected household.

    Args:
        affected_households (pd.DataFrame): A data frame containing the affected households.
        consumption_utility (float): The coefficient of relative risk aversion.
        discount_rate (float): The discount rate.
        average_productivity (float): The average productivity.
        optimization_timestep (float): The timestep for the optimization.
        n_years (int): The number of years in the optimization algorithm.

    Returns:
        pd.DataFrame: A data frame containing the affected households with the recovery rate.
    '''
    
    # Set effective capital stock to zero for renters
    affected_households.loc[affected_households['own_rent']
                            == 'rent', 'keff'] = 0 # V

    # Prepare a data frame to store optimization results
    optimization_results = pd.DataFrame({'aeexp': -1,
                                         'aeexp_house': -1,
                                         'keff': -1,
                                         'v': -1,
                                         'aesav': -1,
                                         'solution': None,
                                         'bankrupt': None}, index=[0])

    optimization_results = optimization_results.reset_index(drop=True).set_index(
        ['aeexp',
         'aeexp_house',
         'keff',
         'v',
         'aesav'])

    # Calculate the recovery rate for each affected household
    affected_households['recovery_rate'] = affected_households['v'].apply(
        lambda x: optimize_recovery_rate(x, optimization_results, consumption_utility, discount_rate, average_productivity, optimization_timestep, n_years))

    # TODO: Check whether this has any impact on anything 
    # optimization_results = optimization_results.sort_index()
    
    return affected_households


def optimize_recovery_rate(x, optimization_results: pd.DataFrame, consumption_utility: float, discount_rate: float, average_productivity: float, optimization_timestep: float, n_years: int) -> float:
    try:
        # Look for the existing solution
        solution = optimization_results.loc[(
            0, 0, 0, round(x, 3), round(average_productivity, 3)), 'solution']
        return solution
    except:
        # No existing solution found, so we need to optimize
        t_max_linspace = n_years  # years
        nsteps_linspace = 52 * t_max_linspace  # total weeks
        dt = t_max_linspace / nsteps_linspace

        _lambda = 0
        opt_step = optimization_timestep
        last_dwdlambda = 0

        while True:

            dwdlambda = 0

            for _t in np.linspace(0, t_max_linspace, nsteps_linspace):

                part1 = average_productivity - \
                    (average_productivity+_lambda)*x*np.e**(-_lambda*_t)
                part1 = part1**(-consumption_utility)

                part2 = _t * (average_productivity+_lambda) - 1

                part3 = np.e**(-_t*(discount_rate+_lambda))

                dwdlambda += part1 * part2 * part3 * dt

            # !: All these do the same
            if last_dwdlambda < 0 and dwdlambda > 0:
                optimization_results.loc[(0, 0, 0, round(x, 3), round(average_productivity, 3)), [
                    'solution', 'bankrupt']] = [_lambda, False]
                optimization_results = optimization_results.sort_index()
                return _lambda
            elif last_dwdlambda > 0 and dwdlambda < 0:
                optimization_results.loc[(0, 0, 0, round(x, 3), round(average_productivity, 3)), [
                    'solution', 'bankrupt']] = [_lambda, False]
                optimization_results = optimization_results.sort_index()
                return _lambda
            
            # !: That's why assigning more than 10 years does not work, we need to change 10 to ??
            elif _lambda > 10:
                optimization_results.loc[(0, 0, 0, round(x, 3), round(average_productivity, 3)), [
                    'solution', 'bankrupt']] = [_lambda, False]
                optimization_results = optimization_results.sort_index()
                return _lambda
            else:
                last_dwdlambda = dwdlambda
            _lambda += opt_step


def integrate_wellbeing(affected_households: pd.DataFrame, 
                        consumption_utility: float, 
                        discount_rate: float, 
                        income_and_expenditure_growth: float, 
                        average_productivity: float, 
                        poverty_line: float, 
                        n_years: int,
                        add_income_loss: bool,
                        cash_transfer: dict = {},
                        ) -> pd.DataFrame:
    
    # We need to reset some columns to zero to start the integration
    columns = ['consumption_loss', 'consumption_loss_NPV', 'net_consumption_loss', 'net_consumption_loss_NPV', 'c_t', 'w_final', 'weeks_pov', 'w_final2']
    affected_households[columns] = [0., 0., 0., 0., 0., 0., 0., 0.]
    
    # Define the number of weeks given the number of years
    n_weeks = 52 * n_years
    dt = n_years / n_weeks
    
    # * Store consumption recovery in a dict for verification and debugging purposes
    # consumption_recovery = {}

    # We need to adjust the cash transfer to the timestep of the integration
    if cash_transfer != {}: # If there is a cash transfer
        cash_transfer_transformed = {np.linspace(0, n_years, n_weeks)[t]: cash_transfer[t] for t in list(cash_transfer.keys())}
    else:
        cash_transfer_transformed = {}

    # Calculate the consumption loss for each affected household
    for _t in np.linspace(0, n_years, n_weeks):
        gfac = (1 + income_and_expenditure_growth)**_t
        
        # TODO: Add an extra condition about to whom the transfer is given
        # A "dynamic policy"
        if _t in list(cash_transfer_transformed.keys()):
            affected_households['aesav'] += cash_transfer_transformed[_t]

        # `c_t` is the consumption at time t
        # !: It seems that keff remains constant over time

        expenditure_growth = gfac * affected_households['aeexp']
        exponential_multiplier = np.e**(-affected_households['recovery_rate']*_t)
        savings = gfac * affected_households['aesav'] * affected_households['recovery_rate']
        asset_damage = gfac * affected_households['v'] * affected_households[['keff', 'recovery_rate']].prod(axis=1)
        income_loss = gfac * (1 - affected_households['delta_tax_safety']) * average_productivity * affected_households['keff'] * affected_households['v']

        # Equation is as follows: consumption_loss = expenditure_growth + exponential_multiplier * (savings - asset_damage - income_loss)

        if add_income_loss:
            affected_households['c_t'] = (expenditure_growth + 
                                          exponential_multiplier * (savings - asset_damage - income_loss))
        else:
            affected_households['c_t'] = (expenditure_growth + 
                                          exponential_multiplier * (savings - asset_damage))
        
        # affected_households['c_t'] = (gfac * affected_households['aeexp']  # expenditure growth
                                    
        #                             + np.e**(-affected_households['recovery_rate']*_t) # exponential multiplier 
                                    
        #                             * (gfac * affected_households['aesav'] * affected_households['recovery_rate'] # savings
        #                                 - gfac * affected_households['v'] * affected_households[['keff', 'recovery_rate']].prod(axis=1))) # asset damage
                                        
        #                                 # - (1 - affected_households['delta_tax_safety']) # income loss
        #                                 #     * average_productivity * affected_households['keff']
        #                                 #     * affected_households['v']))
                
        # `c_t_unaffected` is the consumption at time t if the household was not affected by the disaster
        affected_households['c_t_unaffected'] = gfac * affected_households['aeexp'] 
        
        # TODO: Add a check to see whether the consumption goes below 0

        # Consumption cannot be lower than 0
        affected_households.loc[affected_households['c_t'] < 1, 'c_t'] = 1
        
        # TODO: Add a check whether the consumption after hit by disaster should be lower than or equal to consumption before hit by disaster
        
        # consumption after hit by disaster should be lower than or equal to consumption before hit by disaster
        affected_households.loc[affected_households['c_t'] > affected_households['c_t_unaffected'],
                                'c_t'] = affected_households.loc[affected_households['c_t'] > affected_households['c_t_unaffected'], 'c_t_unaffected']

        # total (integrated) consumption loss
        affected_households['consumption_loss'] += dt * \
            (affected_households['c_t_unaffected'] -
                affected_households['c_t'])
        
        affected_households['consumption_loss_NPV'] += dt * \
            (affected_households['c_t_unaffected'] -
                affected_households['c_t'])*np.e**(-discount_rate*_t)

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
                                < poverty_line, 'weeks_pov'] += 1

        # Integrated wellbeing
        affected_households['w_final'] += dt*(affected_households['c_t'])**(1-consumption_utility) * \
            np.e**(-discount_rate*_t)/(1-consumption_utility)

        # w_final2 version 02
        affected_households['w_final2'] += affected_households['c_t_unaffected']**(1-consumption_utility)/(1-consumption_utility)*dt*(
            (1-((affected_households['c_t_unaffected'] - affected_households['c_t'])/affected_households['c_t_unaffected'])*np.e**(-affected_households['recovery_rate']*_t))**(1-consumption_utility)-1)*np.e**(-discount_rate*_t)

        # * Use to examine individual consumption recovery
        # Save consumption recovery value at the time _t
        # consumption_recovery[_t] = affected_households['c_t']
        
    # Save consumption recovery as pickle file
    # with open('consumption_recovery.pickle', 'wb') as handle:
    #     pickle.dump(consumption_recovery, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return affected_households
