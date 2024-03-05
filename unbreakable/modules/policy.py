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
# TODO: Update module description


import pandas as pd


# TODO: Allow to specify dynamic policies

def cash_transfer(households: pd.DataFrame, current_policy: str) -> pd.DataFrame:
    '''
    Apply cash transfer to a specific target group.

    This function applies a cash transfer policy to a specific target group of households. The policy is specified in the format "<target_group>+<top_up>", where <target_group> can be "all", "poor", "poor_near_poor1.25", or "poor_near_poor2.0", and <top_up> is a percentage value indicating the amount to be added to the 'sav' column of the households.

    Args:
        households (pd.DataFrame): Households.
            Required columns: 'is_affected', 'is_poor', 'exp', 'povline_adjusted', 'keff', 'v', 'sav'
        policy (str): Policy to apply. Format: "<target_group>+<top_up>". Example: "poor+100".

    Returns:
        pd.DataFrame: Households with applied policy.
    '''
    try:
        target_group, top_up = current_policy.split('+')
        top_up = float(top_up)
    except ValueError:
        raise ValueError(
            "policy should be in the format '<target_group>+<top_up>'")

    # Get the adjusted poverty line
    povline_adjusted = households['povline_adjusted'].iloc[0]

    # Filter affected households
    affected_households = households.query('is_affected')

    # Determine beneficiaries based on target group
    if target_group == 'all':
        beneficiaries = affected_households

    elif target_group == 'poor':
        beneficiaries = affected_households.query('is_poor')

        # If there are no poor households, return the original DataFrame
        if len(beneficiaries) == 0:
            return households

    # FIXME: These must be different policies, poor and near poor
    elif target_group in ['poor_near_poor1.25', 'poor_near_poor2.0']:
        multiplier = 1.25 if target_group == 'poor_near_poor1.25' else 2.0
        # Define conditions for poor and near poor households
        poor_condition = affected_households['is_poor'] == True
        near_poor_condition = (~affected_households['is_poor']) & (
            affected_households['exp'] < multiplier * povline_adjusted)

        # Combine conditions to identify beneficiaries
        beneficiary_condition = poor_condition | near_poor_condition

        # If there are no beneficiaries, return the original DataFrame
        if not beneficiary_condition.any():
            return households

        # Select beneficiaries based on the combined condition
        beneficiaries = affected_households.loc[beneficiary_condition]
    else:
        raise ValueError(f"Unknown target group: {target_group}")

    # Apply top-up
    households.loc[beneficiaries.index, 'sav'] += (
        beneficiaries.eval('keff*v') * top_up / 100
    )

    return households


def retrofitting(country: str, households: pd.DataFrame, current_policy: str, random_seed: int) -> pd.DataFrame:
    '''
    Apply retrofitting to a specific target group.

    This function applies a retrofitting policy to a specific target group of households. The policy is specified in the format "<target_group>+<houses_pct>", where <target_group> can be "all", "poor_40", and <houses_pct> is a percentage value indicating the proportion of houses to be retrofitted.

    Args:
        country (str): Country name.
        households (pd.DataFrame): Households.
            Required columns: 'roof', 'walls', 'is_poor', 'exp'
        policy (str): Policy to apply. Format: "<target_group>+<houses_pct>". Example: "poor_40+100".
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Households with applied policy.

    Raises:
        ValueError: If the policy format is incorrect or the households DataFrame does not contain the required columns.   
    '''
    try:
        target_group, houses_pct = current_policy.split('+')
        houses_pct = int(houses_pct)
    except ValueError:
        raise ValueError(
            "my_policy should be in the format '<target_group>+<houses_pct>'")

    # Check whether the households DataFrame contains the required columns
    if not {'roof', 'walls', 'is_poor', 'exp'}.issubset(households.columns):
        raise ValueError(
            "Households DataFrame must contain 'roof', 'walls', 'is_poor', and 'exp' columns")

    # Each country has different retrofitting policies
    if country == 'Dominica':
        # Define which houses are vulnerable (material vulnerability score > 0.5)
        v_roof_types = [
            'Shingle (wood)', 'Shingle (asphalt)', 'Shingle (other)', 'Other']
        v_walls_types = ['Wood/Timber', 'Plywood',
                         'Makeshift', "Other/Don't know", 'Other']
        robust_roof = 'Concrete'
        robust_walls = 'Concrete/Concrete blocks'

        # Find the vulnerable houses where both roof and walls are vulnerable
        vulnerable_houses = households[households['walls'].isin(
            v_walls_types) | households['roof'].isin(v_roof_types)]

        if len(vulnerable_houses) == 0:
            return households

        # Save old roof and walls materials
        households['roof_old'] = households['roof']
        households['walls_old'] = households['walls']

        # Apply the policy to a percentage of the vulnerable houses
        if houses_pct != 100:
            vulnerable_houses = vulnerable_houses.sample(
                frac=houses_pct/100, random_state=random_seed)

        # Search for poor households  if the target group is not 'all'
        if target_group != 'all':
            # Get pct of poor households
            pct_poor = int(target_group.split('_')[1]) / 100

            # Find the poor vulnerable houses
            poor_houses = vulnerable_houses.query('is_poor')

            # Find pct_poor (e.g., 40%) of the poorest expenditure-wise
            poor_houses = poor_houses.nsmallest(
                int(len(poor_houses) * pct_poor), 'exp', keep='all')

            if len(poor_houses) == 0:
                return households

            # Keep only the poor houses
            vulnerable_houses = vulnerable_houses.loc[poor_houses.index]

        # Apply retrofitting to the vulnerable houses
        households['retrofitted'] = False

        # Retrofit roofs
        households.loc[vulnerable_houses.index, 'roof'] = households.loc[vulnerable_houses.index, 'roof'].apply(
            lambda x: robust_roof if x in v_roof_types else x
        )
        # Retrofit walls
        households.loc[vulnerable_houses.index, 'walls'] = households.loc[vulnerable_houses.index, 'walls'].apply(
            lambda x: robust_walls if x in v_walls_types else x
        )
        households.loc[vulnerable_houses.index, 'retrofitted'] = True

        # Calculate the new vulnerability score
        households = recalculate_house_vulnerability(country, households)

        # NOTE: After retrofitting we do not randomize the vulnerability score
        return households

    else:
        raise ValueError(f"Country '{country}' not yet supported")


def recalculate_house_vulnerability(country: str, households: pd.DataFrame) -> pd.DataFrame:
    '''Recalculate vulnerability of a house based on its new retrofitted roof and walls material.

    Args:
        country (str): Country name.
        households (pd.DataFrame): Households.
            Required columns: 'roof', 'walls'

    Returns:
        pd.DataFrame: Households with recalculated vulnerability scores.

    Raises:
        ValueError: If the country is not supported.
    '''
    # Each country has different roof and wall materials and their respective vulnerability scores
    if country == 'Dominica':
        # Define vulnerability scores in dictionaries for easy mapping
        v_roof_scores = {
            'default': 0.75,
            'Concrete': 0.2,
            'Sheet metal (galvanize, galvalume)': 0.35,
            'Shingle (wood)': 0.5,
            'Shingle (asphalt)': 0.5,
            'Shingle (other)': 0.5,
            'Other': 0.75,
        }

        v_walls_scores = {
            'default': 0.6,
            'Wood & Concrete': 0.2,
            'Brick/Blocks': 0.25,
            'Concrete/Concrete blocks': 0.25,
            'Wood/Timber': 0.6,
            'Plywood': 0.7,
            'Makeshift': 0.8,
            "Other/Don't know": 0.8,
            'Other': 0.8,
        }

    else:
        raise ValueError(f"Country '{country}' not yet supported")

    # Apply default scores
    households['v_roof'] = households['roof'].apply(
        lambda x: v_roof_scores.get(x, v_roof_scores['default']))
    households['v_walls'] = households['walls'].apply(
        lambda x: v_walls_scores.get(x, v_walls_scores['default']))

    # Calculate initial vulnerability score
    v_roof_multiplier = 0.3
    v_walls_multiplier = 0.7

    # NOTE: Since it is for a retrofitting policy,
    # we immediately assign the new values to `v` and not `v_init` column
    households['v'] = v_roof_multiplier * \
        households['v_roof'] + v_walls_multiplier * households['v_walls']

    return households


def apply_policy(households: pd.DataFrame, country: str, current_policy: str, random_seed: int) -> pd.DataFrame:
    '''
    Apply a specific policy to a set of households.

    This function applies a specific policy to a set of households, based on their economic status and the impact of external factors. The policy is specified in the format "<policy_type>:<policy_details>", where <policy_type> can be "asp" (cash transfer) or "retrofit" (housing retrofitting), and <policy_details> is a string containing the details of the policy.

    Args:
        country (str): Country name.
        households (pd.DataFrame): Households.
            Required columns: 'is_affected', 'is_poor', 'exp', 'povline_adjusted', 'keff', 'v', 'sav', 'roof', 'walls'
        policy (str): Policy to apply. Format: "<policy_type>:<policy_details>". Example: "asp:poor+100".
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Households with applied policy.

    Raises:
        ValueError: If the policy type is unknown.
    '''
    # Identify the type of policy provided
    policy_type = current_policy.split(':')[0]

    # Apply the policy based on its type
    if policy_type == 'asp':
        return cash_transfer(households, current_policy.split(':')[1])
    elif policy_type == 'retrofit':
        return retrofitting(country, households, current_policy.split(':')[1], random_seed)
    elif policy_type == 'none':
        return households
    else:
        raise ValueError(
            f"Unknown policy type: {policy_type}, please use 'asp', 'retrofit' or 'none'")
