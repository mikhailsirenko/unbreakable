import numpy as np
import pandas as pd
import pickle
import os


def calculate_recovery_rate(households: pd.DataFrame, average_productivity: float, consumption_utility: float, discount_rate: float, lambda_increment: float, years_to_recover: int, is_conflict: bool, conflict: pd.DataFrame = None) -> pd.DataFrame:
    # Assign initial value to recovery_rate
    households['recovery_rate'] = 0

    # Subset households that are affected by the disaster
    affected_households = households[households['is_affected'] == True].copy()

    # If there is a conflict, adjust the average productivity
    if is_conflict:
        region_conflict_intensity = conflict[conflict['region']
                                             == households['region'].values[0]]['conflict_intensity'].values[0]

        adjusted_average_productivity = {'Very high': 0.225,
                                         'High': 0.25,
                                         'Medium': 0.275,
                                         'Low': 0.3,
                                         'Very low': 0.325,
                                         'None': 0.35}
        average_productivity = adjusted_average_productivity[region_conflict_intensity]

    # Search for the recovery rate for each affected household
    affected_households['recovery_rate'] = affected_households['v'].apply(
        lambda x: find_recovery_rate(x, average_productivity, consumption_utility, discount_rate, lambda_increment, years_to_recover))

    assert (affected_households['recovery_rate'] == years_to_recover).any(
    ) == False, 'Recovery rate was not found for one of the households'
    assert (affected_households['recovery_rate'] > 1).any(
    ) == False, 'Recovery rate is greater than 1 for one of the households'
    assert (affected_households['recovery_rate'] == 0).any(
    ) == False, 'Recovery rate is 0 for one of the households'

    households.loc[affected_households.index,
                   'recovery_rate'] = affected_households['recovery_rate']

    return households


def find_recovery_rate(v: float, average_productivity: float, consumption_utility: float, discount_rate: float, lambda_increment: float, years_to_recover: int) -> float:
    totaL_weeks = 52 * years_to_recover
    dt = 1 / 52

    lambda_value = 0
    last_derivative_lambda = 0

    while True:
        derivative_lambda = 0
        for time in np.linspace(0, years_to_recover, totaL_weeks):
            factor = average_productivity + lambda_value
            part1 = (average_productivity - factor * v *
                     np.exp(-lambda_value * time)) ** (-consumption_utility)
            part2 = time * factor - 1
            part3 = np.exp(-time * (discount_rate + lambda_value))
            derivative_lambda += part1 * part2 * part3 * dt

        if (last_derivative_lambda < 0 and derivative_lambda > 0) or (last_derivative_lambda > 0 and derivative_lambda < 0) or lambda_value > years_to_recover:
            return lambda_value

        last_derivative_lambda = derivative_lambda
        lambda_value += lambda_increment


def calculate_wellbeing(households: pd.DataFrame, average_productivity: float, consumption_utility: float, discount_rate: float, income_and_expenditure_growth: float, years_to_recover: int, add_income_loss: bool, save_consumption_recovery: bool, is_conflict: bool = False, conflict: pd.DataFrame = None) -> pd.DataFrame:

    # Get the adjusted poverty line for the region
    poverty_line_adjusted = households['poverty_line_adjusted'].values[0]

    # Add new columns with initial values
    new_columns = ['consumption_loss', 'consumption_loss_npv', 'net_consumption_loss',
                   'net_consumption_loss_npv', 'c_t', 'c_t_unaffected', 'weeks_in_poverty', 'wellbeing']
    households[new_columns] = 0

    # Subset households that are affected by the disaster
    affected_households = households[households['is_affected'] == True].copy()

    # Define the number of weeks given the number of years
    totaL_weeks = 52 * years_to_recover
    dt = 1 / 52

    consumption_recovery = {}

    if is_conflict:
        region_conflict_class = households['conflict_intensity'].values[0]
        marco_multiplier = {'Very high': 0.05,
                            'High': 0.02997,
                            'Medium': 0.0232,
                            'Low': 0.01595,
                            'Very low': 0.00725,
                            'None': 0.0}
        vulnerability_increase_factor = 1 + \
            marco_multiplier[region_conflict_class]
    else:
        vulnerability_increase_factor = 1

    # Integrate consumption loss and well-being
    for time in np.linspace(0, years_to_recover, totaL_weeks):
        exponential_multiplier = np.e**(
            -affected_households['recovery_rate'] * time)

        growth_factor = (1 + income_and_expenditure_growth)**time

        expenditure = growth_factor * affected_households['exp']

        savings = growth_factor * \
            affected_households['sav'] * (1 / vulnerability_increase_factor)

        asset_loss = growth_factor * \
            affected_households['v'] * affected_households['keff'] * \
            affected_households['recovery_rate'] * \
            vulnerability_increase_factor

        income_loss = growth_factor * average_productivity * \
            affected_households['keff'] * \
            affected_households['v'] * vulnerability_increase_factor

        # asset_loss = growth_factor * affected_households['v'] * \
        #     (affected_households['exp_house'] +
        #      affected_households[['keff', 'recovery_rate']].prod(axis=1))

        if add_income_loss == False:
            affected_households['c_t'] = (expenditure +
                                          exponential_multiplier * (savings - asset_loss))
        else:
            affected_households['c_t'] = (expenditure +
                                          exponential_multiplier * (savings - asset_loss - income_loss))

        affected_households['c_t_unaffected'] = expenditure

        # Avoid negative consumption
        if (affected_households['c_t'] < 0).any():
            affected_households.loc[affected_households['c_t'] < 0, 'c_t'] = 0

        # Make sure that consumption does not exceed the unaffected consumption
        if (affected_households['c_t'] > affected_households['c_t_unaffected']).any():
            affected_households.loc[affected_households['c_t'] > affected_households['c_t_unaffected'],
                                    'c_t'] = affected_households.loc[affected_households['c_t'] > affected_households['c_t_unaffected'], 'c_t_unaffected']

        # Calculate consumption loss
        affected_households['consumption_loss'] += dt * \
            (affected_households['c_t_unaffected'] -
                affected_households['c_t'])

        affected_households['consumption_loss_npv'] += dt * \
            (affected_households['c_t_unaffected'] -
                affected_households['c_t'])*np.e**(-discount_rate*time)

        # BUG: That's wrong, if exp_house = 0 then then net_consumption_loss is 0
        affected_households['net_consumption_loss'] += dt * \
            np.e**(-affected_households['recovery_rate']*time) * \
            affected_households['v'] * growth_factor * \
            affected_households['exp_house']

        # BUG: That's wrong, if exp_house = 0 then then net_consumption_loss_npv is 0
        affected_households['net_consumption_loss_npv'] += dt * \
            np.e**(-affected_households['recovery_rate']*time) * affected_households['v'] * growth_factor * \
            affected_households['exp_house'] * \
            np.e**(-discount_rate*time)

        # Increase the number of weeks in poverty
        affected_households.loc[affected_households['c_t']
                                < poverty_line_adjusted, 'weeks_in_poverty'] += 1

        # Calculate wellbeing
        affected_households['wellbeing'] += affected_households['c_t_unaffected']**(1 - consumption_utility)\
            / (1 - consumption_utility) * dt \
            * ((1 - ((affected_households['c_t_unaffected'] - affected_households['c_t']) / affected_households['c_t_unaffected'])
                * np.e**(-affected_households['recovery_rate'] * time))**(1 - consumption_utility) - 1)\
            * np.e**(-discount_rate * time)

        # Store the consumption recovery for each time step
        if save_consumption_recovery:
            consumption_recovery[time] = affected_households.loc[:, [
                'wgt', 'is_poor', 'recovery_rate', 'c_t_unaffected', 'c_t']].set_index('wgt')

    # Save the content of the columns for affected_households into households data frame
    households.loc[affected_households.index,
                   new_columns] = affected_households[new_columns]

    # Save consumption recovery as a pickle file
    if save_consumption_recovery:
        country = households['country'].values[0]
        region = households['region'].values[0]
        return_period = households['return_period'].values[0]
        try:
            random_seed = households['random_seed'].values[0]
        except:
            random_seed = None

        save_consumption_recovery_to_file(consumption_recovery,
                                          country, region, return_period, is_conflict, random_seed)

    return households


def save_consumption_recovery_to_file(consumption_recovery: dict, country: str, region: str, return_period: str, is_conflict: bool, random_seed: int):
    if is_conflict:
        folder = f'../experiments/{country}/consumption_recovery/return_period={return_period}/conflict={is_conflict}/{random_seed}'
    else:
        folder = f'../experiments/{country}/consumption_recovery/return_period={return_period}/conflict={is_conflict}/{random_seed}'

    # Create a folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save consumption recovery as pickle file
    with open(folder + f'/{region}.pickle', 'wb') as handle:
        pickle.dump(consumption_recovery, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
