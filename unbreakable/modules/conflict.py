import pandas as pd


def affect_economy(conflict_impact: pd.DataFrame, avg_prod: float, inc_exp_growth: float) -> pd.DataFrame:
    '''Calculate the effect of conflict on average productivity and income and expenditure growth.

    Args:
        conflict_impact (pd.DataFrame): Conflict impact data.
        avg_prod (float): Average productivity.
        inc_exp_growth (float): Income and expenditure growth.

    Returns:
        pd.DataFrame: Affected economy.
    '''
    # Make a copy of the conflict impact data
    df = conflict_impact.copy()

    # Decrease the average productivity by Avg. prod. decrease %
    df['avg_prod'] = avg_prod - \
        df['Avg. prod. decrease'] * avg_prod

    # Decrease the income and expenditure growth by Inc. exp. growth decrease %
    df['inc_exp_growth'] = inc_exp_growth - \
        df['Inc. and exp. growth decrease'] * inc_exp_growth

    # Return affected economy
    return df
