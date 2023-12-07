import pandas as pd


def apply_policy(households: pd.DataFrame, my_policy: str) -> pd.DataFrame:
    # TODO: Move into a separate module
    '''Apply a policy to a specific target group.

    Args:
        households (pd.DataFrame): Households.
        my_policy (str): Policy to apply. Format: <target_group>+<top_up>. Example: "poor+100".
    Returns:
        pd.DataFrame: Households with applied policy.
    '''
    # * Note that we adjusted the poverty line in `adjust_assets_and_expenditure`
    poverty_line_adjusted = households['poverty_line_adjusted'].iloc[0]

    target_group, top_up = my_policy.split('+')
    top_up = float(top_up)

    if target_group == 'all':
        beneficiaries = households['is_affected'] == True

    elif target_group == 'poor':
        beneficiaries = (households['is_affected'] == True) & (
            households['is_poor'] == True)

    elif target_group == 'poor_near_poor1.25':
        poor_affected = (households['is_affected'] == True) & (
            households['is_poor'] == True)
        near_poor_affected = (households['is_affected'] == True) & (
            households['is_poor'] == False) & (households['aeexp'] < 1.25 * poverty_line_adjusted)
        beneficiaries = poor_affected | near_poor_affected

    elif target_group == 'poor_near_poor2.0':
        poor_affected = (households['is_affected'] == True) & (
            households['is_poor'] == True)
        near_poor_affected = (households['is_affected'] == True) & (
            households['is_poor'] == False) & (households['aeexp'] < 2 * poverty_line_adjusted)
        beneficiaries = poor_affected | near_poor_affected

    households.loc[beneficiaries,
                   'aesav'] += households.loc[beneficiaries].eval('keff*v') * top_up / 100

    return households
