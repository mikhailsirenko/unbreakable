import numpy as np
import pandas as pd


def generate_households(num_households=1):
    '''Generate dummy households.'''
    np.random.seed(0)  # Ensure reproducibility
    data = {
        'exp': np.random.uniform(1000, 5000, num_households),
        'sav': np.random.uniform(100, 1000, num_households),
        'v': np.random.uniform(0.2, 0.8, num_households),
        'keff': np.random.uniform(500, 15000, num_households),
        'exp_house': 0,
        'consumption_loss': 0,
        'consumption_loss_npv': 0,
        'net_consumption_loss': 0,
        'net_consumption_loss_npv': 0,
        'c_t': 0,
        'c_t_unaffected': 0,
        'wellbeing': 0
    }
    return pd.DataFrame(data)


def generate_risk_and_damage():
    pass


def generate_conflict():
    pass
