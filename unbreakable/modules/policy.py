"""This module provides a function to apply a specific monetary policy to a set of households, based on their economic status and the impact of external factors.

The main function in this module is `apply_policy`, which takes a DataFrame representing households and a policy string as input. The policy is defined by a target group and a top-up percentage, indicating the increase in savings or assets for the selected households. The target group can be 'all', 'poor', 'poor_near_poor1.25', or 'poor_near_poor2.0'. The function modifies the 'aesav' column of the DataFrame, representing the adjusted savings or assets after applying the policy.

Example usage:
    import pandas as pd
    from your_module_name import apply_policy

    # Load household data
    households_data = pd.read_csv('households.csv')

    # Define a policy to apply - for instance, increasing assets by 100% for poor households
    policy = 'poor+100'

    # Apply the policy to the household data
    updated_households = apply_policy(households_data, policy)

    # The updated_households DataFrame now contains the adjusted 'aesav' values for the targeted households
"""


import pandas as pd


# TODO: Allow to specify dynamic policies


def apply_policy(households: pd.DataFrame, my_policy: str) -> pd.DataFrame:
    '''
    Apply a policy to a specific target group.

    This function applies a policy to a specific target group of households. The policy is specified in the format "<target_group>+<top_up>", where <target_group> can be "all", "poor", "poor_near_poor1.25", or "poor_near_poor2.0", and <top_up> is a percentage value indicating the amount to be added to the 'aesav' column of the households.

    Args:
        households (pd.DataFrame): Households.
            Required columns: 'is_affected', 'is_poor', 'aeexp', 'povline_adjusted', 'keff', 'v', 'aesav'
        my_policy (str): Policy to apply. Format: "<target_group>+<top_up>". Example: "poor+100".

    Returns:
        pd.DataFrame: Households with applied policy.
    '''
    try:
        target_group, top_up = my_policy.split('+')
        top_up = float(top_up)
    except ValueError:
        raise ValueError(
            "my_policy should be in the format '<target_group>+<top_up>'")

    # Get the adjusted poverty line
    povline_adjusted = households['povline_adjusted'].iloc[0]

    # Filter affected households
    # !: Seems to be wrong
    affected_households = households.query('is_affected')

    # Determine beneficiaries based on target group
    if target_group == 'all':
        beneficiaries = affected_households

    elif target_group == 'poor':
        beneficiaries = affected_households.query('is_poor')

    elif target_group in ['poor_near_poor1.25', 'poor_near_poor2.0']:
        multiplier = 1.25 if target_group == 'poor_near_poor1.25' else 2.0
        condition = f'is_poor or (not is_poor and aeexp < {multiplier} * @povline_adjusted)'
        beneficiaries = affected_households.query(condition)

    else:
        raise ValueError(f"Unknown target group: {target_group}")

    # Apply top-up
    households.loc[beneficiaries.index, 'aesav'] += (
        beneficiaries.eval('keff*v') * top_up / 100
    )

    return households
