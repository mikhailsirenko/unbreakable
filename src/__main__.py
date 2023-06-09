from model import Model
# Use this file to run the model from the command line.

# Inputs:
# * parameters;
# * asset damage data;
# * household survey data.
# * must have columns
# *

# Outputs:
# * results;
# * Must have columns

country = 'Saint Lucia'
districts = ['Anse-la-Raye',
             'Canaries',
             'Castries',
             'Choiseul',
             'Dennery',
             'Gros Islet',
             'Laborie',
             'Micoud',
             'Soufriϋre',
             'Vieux Fort']
scale = 'district'

read_parameters_from_file = False

constants = {'poverty_line': 960,
             'indigence_line': 288,
             'saving_rate': 0.75}

uncertainties = {'income_and_expenditure_growth': 0.02,
                 'poverty_bias': 1.0,
                 'discount_rate': 0.06,
                 'consumption_utility': 1.5,
                 'is_vulnerability_random': False,
                 'adjust_assets_and_expenditure': True,
                 'min_households': 1493}

simulation = {'n_replications': 5,
              'optimization_timestep': 0.01}

scenarios = [{'return_period': 100}]

policies = [{'': 'None'}]

parameters = {'country': country,
              'scale': scale,
              'constants': constants,
              'uncertainties': uncertainties,
              'simulation': simulation}

if __name__ == "__main__":
    for policy in policies:
        parameters['policy'] = [
            key for key in policy.values()][0]  # current policy
        for scenario in scenarios:
            for district in districts:
                parameters['district'] = district
                parameters['scenario'] = [
                    key for key in scenario.values()][0]  # current scenario
                my_model = Model(print_parameters=True, **parameters)
                my_model.run_simulation()
