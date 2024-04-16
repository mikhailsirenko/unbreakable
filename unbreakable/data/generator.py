import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats


def generate_households(num_households=1, conflict_intensity: str = 'None', fix_seed=True):
    '''Generate dummy households.'''
    if fix_seed:
        np.random.seed(0)  # Ensure reproducibility

    data = {'exp_house': 0,
            'consumption_loss': 0,
            'consumption_loss_npv': 0,
            'net_consumption_loss': 0,
            'net_consumption_loss_npv': 0,
            'c_t': 0,
            'c_t_unaffected': 0,
            'wellbeing': 0}

    if conflict_intensity != 'None':
        d = {'Very low': {'mean': 1522.72, 'std': 980.51, 'min': 154.25, 'max': 5473.81},
             'Low': {'mean': 1333.83, 'std': 799.99, 'min': 226.35, 'max': 5439.8},
             'Medium': {'mean': 982.08, 'std': 666.34, 'min': 175.05, 'max': 5317.89},
             'High': {'mean': 1064.61, 'std': 809.55, 'min': 156.39, 'max': 5439.94},
             'Very high': {'mean': 637.02, 'std': 474.87, 'min': 152.8, 'max': 5172.65}}

        lower, upper = d[conflict_intensity]['min'], d[conflict_intensity]['max']
        mu, sigma = d[conflict_intensity]['mean'], d[conflict_intensity]['std']
        X = stats.truncnorm((lower - mu) / sigma,
                            (upper - mu) / sigma, loc=mu, scale=sigma)
        exp = X.rvs(num_households)
    else:
        # Let's take q=0.25 and q=0.75 as the lower and upper bounds
        lower, upper = 153, 5474
        mu, sigma = 1099, 1099
        X = stats.truncnorm((lower - mu) / sigma,
                            (upper - mu) / sigma, loc=mu, scale=sigma)
        exp = X.rvs(num_households)

    data['exp'] = exp
    # Income is a product of expenditure and a random coefficient
    inc_multiplier = 1.48  # for Nigeria
    inc_delta = 0.1
    low = inc_multiplier - inc_delta
    high = inc_multiplier + inc_delta
    data['inc'] = data['exp'] * np.random.uniform(low, high)

    sav_multiplier = 0.0204  # for Nigeria
    sav_delta = 0.02
    low = sav_multiplier - sav_delta
    high = sav_multiplier + sav_delta
    data['sav'] = data['inc'] * np.random.uniform(low, high)

    # divide by average productivity, 0.35 for Nigeria
    data['keff'] = data['inc'] / 0.35

    mean_vulnerability = {'Very low': 0.43,
                          'Low': 0.46,
                          'Medium': 0.66,
                          'High': 0.63,
                          'Very high': 0.65}

    if conflict_intensity != 'None':
        data['v'] = mean_vulnerability[conflict_intensity]
    else:
        data['v'] = np.random.uniform(0.2, 0.8, num_households)

    # Sort columns
    sorted_columns = ['exp', 'inc', 'sav', 'keff', 'exp_house', 'v', 'consumption_loss', 'consumption_loss_npv',
                      'net_consumption_loss', 'net_consumption_loss_npv', 'c_t', 'c_t_unaffected', 'wellbeing']

    return pd.DataFrame(data)[sorted_columns]


def generate_risk_and_damage():
    pass


def generate_conflict():
    pass
