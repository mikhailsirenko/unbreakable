# TODO: Check how v_init was computed in prepare_data.py

from model import *
from ema_workbench import (Model, Constant, CategoricalParameter, IntegerParameter, RealParameter, ArrayOutcome, MultiprocessingEvaluator, ema_logging, perform_experiments, save_results)

ema_logging.log_to_stderr(ema_logging.INFO)

my_model = Model(name='model', function=run_model)

if __name__ == '__main__':
    country = 'Saint Lucia'
    scale = 'district'
    districts = ['AnseLaRayeCanaries', 'Castries', 'Choiseul', 'Dennery', 'Gros Islet', 'Laborie', 'Micoud', 'Soufriere', 'Vieuxfort']

    kwargs = {
        # Case study constants
        'return_period': 100,
        'poverty_line': 6443,  # EC$ per capita per year
        'indigence_line': 2123,  # EC$ per capita per year
        'saving_rate': 0.02385,
        'is_vulnerability_random': False,
        'min_households': 1493, # min households we need to have a good enough representation
        'optimization_timestep': 0.01,
        'x_max': 10,  # number of years in optimization algorithm

        # Uncertainties
        'poverty_bias': 1.0,
        'consumption_utility': 1.0,
        'discount_rate': 0.04,
        'income_and_expenditure_growth': 0.01,

        # Policy levers
        'my_policy': 'None',

        # Model constants
        "assign_savings_params": {
            "mean_noise_low": 0,
            "mean_noise_high": 5,
            "mean_noise_distribution": "uniform",
            "noise_scale": 2.5,
            "noise_distribution": "normal",
            "savings_clip_min": 0.1,
            "savings_clip_max": 1.0
        },
        "set_vulnerability_params": {
            "vulnerability_random_low": 0.01,
            "vulnerability_random_high": 0.9,
            "vulnerability_random_distribution": "uniform",
            "vulnerability_initial_low": 0.6,
            "vulnerability_initial_high": 1.4,
            "vulnerability_initial_distribution": "uniform",
            "vulnerability_initial_threshold": 0.95
        },
        "calculate_exposure_params": {
            "poverty_bias_random_distribution": "uniform",
            "poverty_bias_random_low": 0.5,
            "poverty_bias_random_high": 1.5
        },
        "determine_affected_params": {
            "low": 0,
            "high": 1.0,
            "distribution": "uniform",
            "delta_pct" : 0.025,
            "num_masks": 1000,
        }
    }

    seed_start = 0
    seed_end = 1000000

    my_model.constants = [# Case study constants
                          Constant('country', country),
                          Constant('scale', scale),
                          Constant('districts', districts),
                          Constant('print_statistics', False),
                          Constant('return_period', kwargs['return_period']),
                          Constant('poverty_line', kwargs['poverty_line']),
                          Constant('indigence_line', kwargs['indigence_line']),
                          Constant('saving_rate', kwargs['saving_rate']),
                          Constant('is_vulnerability_random', kwargs['is_vulnerability_random']),
                          Constant('min_households', kwargs['min_households']),
                          Constant('optimization_timestep', kwargs['optimization_timestep']),
                          Constant('x_max', kwargs['x_max']),

                          # Uncertainties
                          Constant('poverty_bias', kwargs['poverty_bias']),
                          Constant('consumption_utility', kwargs['consumption_utility']),
                          Constant('discount_rate', kwargs['discount_rate']),

                          # Policy levers
                          Constant('my_policy', kwargs['my_policy']),
                          
                          # Model constants
                          Constant('income_and_expenditure_growth', kwargs['income_and_expenditure_growth']),
                          Constant('assign_savings_params', kwargs['assign_savings_params']),
                          Constant('set_vulnerability_params', kwargs['set_vulnerability_params']),
                          Constant('calculate_exposure_params', kwargs['calculate_exposure_params']),
                          Constant('determine_affected_params', kwargs['determine_affected_params'])]

    my_model.uncertainties = [
                              IntegerParameter("random_seed", seed_start, seed_end),
    #                         RealParameter('poverty_bias', 1.0, 1.5), # 1.0, 1.5
    #                         RealParameter('consumption_utility', 1.0, 1.5), # 1.0, 1.5
    #                         RealParameter('discount_rate', 0.04, 0.07), # 0.04, 0.07
    #                         RealParameter('income_and_expenditure_growth', 0.01, 0.03)] # 0.01, 0.03
                             ]

    my_model.levers = [
                       CategoricalParameter('top_up', [0, 10, 30, 50]),
                       CategoricalParameter('target_group', ['all', 'poor', 'poor_near_poor1.25', 'poor_near_poor2.0'])]

    my_model.outcomes = [
                         ArrayOutcome('AnseLaRayeCanaries'),
                         ArrayOutcome('Castries'),
                         ArrayOutcome('Choiseul'),
                         ArrayOutcome('Dennery'),
                         ArrayOutcome('Gros Islet'),
                         ArrayOutcome('Laborie'),
                         ArrayOutcome('Micoud'),
                         ArrayOutcome('Soufriere'),
                         ArrayOutcome('Vieuxfort')
                         ]
    
    results = perform_experiments(
        models=my_model, scenarios=2, policies=2)
    
    n_scenarios = 100
    n_policies = 100
    with MultiprocessingEvaluator(my_model) as evaluator:
        results = evaluator.perform_experiments(scenarios=n_scenarios, 
                                                policies=n_policies
                                                )

    # Save results as tar.gz file
    save_results(results, f'../results/scenarios={n_scenarios}, policies={n_policies}.tar.gz')