import numpy as np
import pandas as pd


def run_optimization(affected_households: pd.DataFrame, consumption_utility: float, discount_rate: float, average_productivity: float, optimization_timestep: float) -> pd.DataFrame:
    # Set effective capital stock to zero for renters
    affected_households.loc[affected_households['own_rent']
                            == 'rent', 'keff'] = 0

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

    optimization_results = optimization_results.sort_index()

    # Calculate the recovery rate for each affected household
    affected_households['recovery_rate'] = affected_households['v'].apply(
        lambda x: optimize_recovery_rate(x, optimization_results, consumption_utility, discount_rate, average_productivity, optimization_timestep))

    return affected_households


def optimize_recovery_rate(x, optimization_results: pd.DataFrame, consumption_utility: float, discount_rate: float, average_productivity: float, optimization_timestep: float) -> float:
    try:
        # Look for the existing solution
        solution = optimization_results.loc[(
            0, 0, 0, round(x, 3), round(average_productivity, 3)), 'solution']
        return solution
    except:
        # No existing solution found, so we need to optimize
        t_max_linspace = 10  # years
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
            elif _lambda > 10:
                optimization_results.loc[(0, 0, 0, round(x, 3), round(average_productivity, 3)), [
                    'solution', 'bankrupt']] = [_lambda, False]
                optimization_results = optimization_results.sort_index()
                return _lambda
            else:
                last_dwdlambda = dwdlambda
            _lambda += opt_step


def integrate_wellbeing(affected_households: pd.DataFrame, consumption_utility: float, discount_rate: float, income_and_expenditure_growth: float, average_productivity: float, poverty_line: float, x_max: int) -> pd.DataFrame:
    # TODO: Add docstring
    # TODO: Document the assumptions of this function
    # TODO: Do not hard code the parameters here. Move it to a config file.
    # TODO: Extensive testing

    # household_consumption = {}

    # affected_households['w_init'] = (affected_households['keff']*average_productivity+df_aff['aesoc'])**(1-consumption_utility)/(discount_rate*(1-consumption_utility))
    affected_households[['consumption_loss',
                         'consumption_loss_NPV',
                         'net_consumption_loss',
                         'net_consumption_loss_NPV']] = [0., 0., 0., 0.]

    tstep_n = 52 * x_max  # total weeks
    dt = x_max / tstep_n

    affected_households[['c_t', 'w_final',
                                'weeks_pov', 'w_final2']] = [0, 0, 0, 0]
    for _t in np.linspace(0, x_max, tstep_n):
        # ? What does it mean?
        # TODO: dynamic policy e.g. cash transfer at time t

        # print(poverty_line, indigence_line)

        gfac = (1 + income_and_expenditure_growth)**_t

        # consumption at this timestep
        affected_households['c_t'] = (affected_households['aeexp'] * gfac
                                      + np.e**(-affected_households['recovery_rate']*_t)
                                      * (affected_households['aesav'] * affected_households['recovery_rate']
                                         - affected_households['v'] * gfac
                                         * (affected_households['aeexp_house']
                                            + affected_households[['keff', 'recovery_rate']].prod(axis=1))
                                         - (1-affected_households['delta_tax_safety'])
                                         * average_productivity*affected_households['keff']
                                         * affected_households['v']))
        affected_households['c_t_unaffected'] = affected_households['aeexp']*gfac
        # consumption cannot be lower than 0
        affected_households.loc[affected_households['c_t']
                                < 1, 'c_t'] = 1
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

        # weeks in poverty
        affected_households.loc[affected_households['c_t']
                                < poverty_line, 'weeks_pov'] += 1

        # integrated wellbeing
        affected_households['w_final'] += dt*(affected_households['c_t'])**(1-consumption_utility) * \
            np.e**(-discount_rate*_t)/(1-consumption_utility)

        # w_final2 version 01 - there is a bug
        # affected_households['w_final2'] += affected_households['c_t_unaffected']**(1-consumption_utility)/(1-consumption_utility)*dt*((1-(affected_households['consumption_loss']/affected_households['c_t_unaffected'])*np.e**(-affected_households['recovery_rate']*_t))**(1-consumption_utility)-1)*np.e**(-discount_rate*_t)

        # w_final2 version 02
        affected_households['w_final2'] += affected_households['c_t_unaffected']**(1-consumption_utility)/(1-consumption_utility)*dt*(
            (1-((affected_households['c_t_unaffected'] - affected_households['c_t'])/affected_households['c_t_unaffected'])*np.e**(-affected_households['recovery_rate']*_t))**(1-consumption_utility)-1)*np.e**(-discount_rate*_t)

        # household_consumption[_t] = affected_households['c_t'].values
    # return household_consumption
    return affected_households