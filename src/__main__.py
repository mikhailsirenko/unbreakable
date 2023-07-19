from model import SimulationModel
from ema_workbench import (Model,
                           ScalarOutcome,
                           Constant,
                           IntegerParameter,
                           RealParameter,
                           CategoricalParameter,
                           ArrayOutcome,
                           MultiprocessingEvaluator,
                           ema_logging,
                           perform_experiments,
                           save_results)

ema_logging.log_to_stderr(ema_logging.INFO)

country = 'Saint Lucia'
scale = 'district'
districts = [
    'AnseLaRayeCanaries',
    'Castries',
    'Choiseul',
    'Dennery',
    'Gros Islet',
    'Laborie',
    'Micoud',
    'Soufriere',
    'Vieuxfort'
]

# kwargs play the role of constants
kwargs = {
    # Case study constants
    'return_period': 100,
    'poverty_line': 6443,  # EC$ per capita per year
    'indigence_line': 2123,  # EC$ per capita per year
    'saving_rate': 0.02385,
    'is_vulnerability_random': False,
    # Min number of households that we need to have in the household survey
    'min_households': 1493,
    'optimization_timestep': 0.01,
    'x_max': 10,  # Number of years in optimization algorithm

    # Uncertainty constants
    'poverty_bias': 1.0,
    'consumption_utility': 1.0,
    'discount_rate': 0.04,
    'income_and_expenditure_growth': 0.01,

    # TODO: Inspect/test/sensitivity analysis of model constants
    # Model constants
    "_assign_savings_params": {
        "mean_noise_low": 0,
        "mean_noise_high": 5,
        "mean_noise_distribution": "uniform",
        "noise_scale": 2.5,
        "noise_distribution": "normal",
        "savings_clip_min": 0.1,
        "savings_clip_max": 1.0
    },
    "_set_vulnerability_params": {
        "vulnerability_random_low": 0.01,
        "vulnerability_random_high": 0.9,
        "vulnerability_random_distribution": "uniform",
        "vulnerability_initial_low": 0.6,
        "vulnerability_initial_high": 1.4,
        "vulnerability_initial_distribution": "uniform",
        "vulnerability_initial_threshold": 0.95
    },
    "_calculate_exposure_params": {
        "poverty_bias_random_distribution": "uniform",
        "poverty_bias_random_low": 0.5,
        "poverty_bias_random_high": 1.5
    },
    "_determine_affected_params": {
        "low": 0,
        "high": 0.5,
        "distribution": "uniform"
    },

    "_apply_individual_policy_params": {
        "retrofit_a": 0.05,
        "retrofit_b": 0.7,
        "retrofit_c": 0.2,
        "retrofit_clip_lower": 0,
        "retrofit_clip_upper": 0.7,
        'retrofit_roof1_roof_materials_of_interest' : [2, 4, 5, 6],
        'retrofit_roof1_a' : 0.05,
        'retrofit_roof1_b' : 0.1,
        'retrofit_roof1_c' : 0.2,
        'retrofit_roof1_d' : 0.1,}
}

my_model = SimulationModel(
    country=country, scale=scale, districts=districts, **kwargs)
my_model = Model(name='model', function=my_model.run_model)

# TODO: Check how v_init was computed in prepare_data.py
n_scenarios = 100
seed_start = 0
seend_end = 1000000

my_model.uncertainties = [
                            IntegerParameter("random_seed", seed_start, seend_end),
#                           RealParameter('poverty_bias', 1.0, 1.5), # 1.0, 1.5
#                           RealParameter('consumption_utility', 1.0, 1.5), # 1.0, 1.5
#                           RealParameter('discount_rate', 0.04, 0.07), # 0.04, 0.07
#                           RealParameter('income_and_expenditure_growth', 0.01, 0.03)] # 0.01, 0.03
                         ]

# my_model.levers = [CategoricalParameter('my_policy', ['None', 'PDS'])]

my_model.outcomes = [ArrayOutcome('AnseLaRayeCanaries'),
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
    models=my_model, scenarios=n_scenarios)

# Save results as tar.gz file
high = kwargs['_determine_affected_params']['high']
save_results(results, f'../results/results_{n_scenarios}_high={high}.tar.gz')

# with MultiprocessingEvaluator(my_model) as evaluator:
#     results = evaluator.perform_experiments(scenarios=2)s