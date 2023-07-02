from model import Model
# Use this file to run the model from the command line.

# TODO: Combine Anse-la-Raye and Canaries in the asset damage data

country = 'Saint Lucia'
# * There is a mismatch between the names of the districts in the household survey and the names of the districts in asset damage data.
districts = [
              'AnseLaRayeCanaries',  # <- 'Anse-la-Raye' + 'Canaries'
             # 'Canaries',  # V <- 'Canaries' + 'Anse-la-Raye'
              'Castries',  # V
              'Choiseul',  # V
              'Dennery',  # V   
              'Gros Islet',  # V
              'Laborie',  # V
              'Micoud',  # V
              'Soufriere',  # <- 'Soufriϋre',
              'Vieuxfort'  # <- 'Vieux Fort'
              ]  

scale = 'district'

read_parameters_from_file = False

constants = {'poverty_line': 6443, # EC$ per capita per year
             'indigence_line': 2123, # EC$ per capita per year
             'saving_rate': 0.02385}

uncertainties = {'income_and_expenditure_growth': 0.02, # 0.01 - 0.03
                 'poverty_bias': 1.0, # 
                 'discount_rate': 0.06, # 0.04 - 0.07
                 'consumption_utility': 1.5,
                 'is_vulnerability_random': False,
                 'adjust_assets_and_expenditure': True,
                 'min_households': 1493}

simulation = {'n_replications': 200,
              'optimization_timestep': 0.01}

scenarios = [{'return_period': 100}]

available_policies = ['Existing_SP_100', 'Existing_SP_50',
                      'retrofit', 'retrofit_roof1', 'PDS', 'None']
policies = [{ '': 'None'}, 
            # {'' : 'PDS'},
            # {'' : 'retrofit'},
            # {'' : 'retrofit_roof1'},
            # {'' : 'Existing_SP_50'},
            # {'' : 'Existing_SP_100'}
            ]

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
                my_model = Model(**parameters)
                my_model.run_simulation()