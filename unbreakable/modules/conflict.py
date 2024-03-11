import pandas as pd


def calculate_conflict_impact(conflict: pd.DataFrame, avg_prod: float, inc_exp_growth: float) -> tuple:
    '''Calculate the effect of a conflict on average productivity and income and expenditure growth by region.

    Args:
        conflict (pd.DataFrame): Conflict data.
        avg_prod (float): Base average productivity.
        inc_exp_growth (float): Base income and expenditure growth.

    Returns:
        pd.DataFrame: Affected average productivity and income and expenditure growth.
    '''
    # Decrease the average productivity by Avg. prod. decrease %
    conflict['avg_prod'] = avg_prod - \
        conflict['Avg. prod. decrease'] * avg_prod

    # Decrease the income and expenditure growth by Inc. exp. growth decrease %
    conflict['inc_exp_growth'] = inc_exp_growth - \
        conflict['Inc. and exp. growth decrease'] * inc_exp_growth

    return conflict.set_index('region')['avg_prod'], conflict.set_index('region')['inc_exp_growth']
