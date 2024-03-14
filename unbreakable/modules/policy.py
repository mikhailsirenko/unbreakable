import pandas as pd
import numpy as np


def cash_transfer(households: pd.DataFrame, current_policy: str) -> pd.DataFrame:
    '''
    Apply cash transfer to a specific target group.

    This function applies a cash transfer policy to a specific target group of households. The policy is specified in the format "<target_group>+<top_up>", where <target_group> can be "all", "poor", "poor_near_poor1.25", or "poor_near_poor2.0", and <top_up> is a percentage value indicating the amount to be added to the 'sav' column of the households.

    Args:
        households (pd.DataFrame): Households.
            Required columns: 'is_affected', 'is_poor', 'exp', 'poverty_line_adjusted', 'keff', 'v', 'sav'
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
    poverty_line_adjusted = households['poverty_line_adjusted'].iloc[0]

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

    # TODO: Split into separate policies poor and near_poor
    elif target_group in ['poor_near_poor1.25', 'poor_near_poor2.0']:
        multiplier = 1.25 if target_group == 'poor_near_poor1.25' else 2.0
        # Define conditions for poor and near poor households
        poor_condition = affected_households['is_poor'] == True
        near_poor_condition = (~affected_households['is_poor']) & (
            affected_households['exp'] < multiplier * poverty_line_adjusted)

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


def retrofitting(country: str, households: pd.DataFrame, current_policy: str, disaster_type: str, random_seed: int) -> pd.DataFrame:
    '''
    Apply retrofitting to a specific target group.

    This function applies a retrofitting policy to a specific target group of households. The policy is specified in the format "<target_group>+<houses_pct>", where <target_group> can be "all", "poor_40", and <houses_pct> is a percentage value indicating the proportion of houses to be retrofitted.

    Args:
        country (str): Country name.
        households (pd.DataFrame): Households.
            Required columns: 'roof', 'walls', 'is_poor', 'exp'
        policy (str): Policy to apply. Format: "<target_group>+<houses_pct>". Example: "poor_40+100".
        disaster_type (str): Type of disaster. Example: "hurricane".
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Households with applied policy.

    Raises:
        ValueError: If the policy format is incorrect or the households DataFrame does not contain the required columns.   
    '''
    if random_seed is not None:
        # Set random seed for reproducibility
        np.random.seed(random_seed)

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

        # Select only the ones that own
        vulnerable_houses = vulnerable_houses[vulnerable_houses['own_rent'] == 'own']

        if len(vulnerable_houses) == 0:
            return households

        # Save old roof and walls materials
        households['roof_old'] = households['roof']
        households['walls_old'] = households['walls']

        # Apply the policy to a percentage of the vulnerable houses
        if houses_pct != 100:
            if random_seed is not None:
                # ?: Check if fixing the np random seed affects the sampling
                vulnerable_houses = vulnerable_houses.sample(
                    frac=houses_pct/100, random_state=random_seed)
            else:
                vulnerable_houses = vulnerable_houses.sample(
                    frac=houses_pct/100)

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
            vulnerable_houses = vulnerable_houses.loc[poor_houses.  index]

        # Apply retrofitting to the vulnerable houses
        households['retrofitted'] = False

        # Retrofit roofs
        # current_roofs = households.loc[vulnerable_houses.index, 'roof']
        # current_walls = households.loc[vulnerable_houses.index, 'walls']

        households.loc[vulnerable_houses.index, 'roof'] = households.loc[vulnerable_houses.index, 'roof'].apply(
            lambda x: robust_roof if x in v_roof_types else x
        )
        # Retrofit walls
        households.loc[vulnerable_houses.index, 'walls'] = households.loc[vulnerable_houses.index, 'walls'].apply(
            lambda x: robust_walls if x in v_walls_types else x
        )

        # retrofitted_roofs = households.loc[vulnerable_houses.index, 'roof']
        # retrofitted_walls = households.loc[vulnerable_houses.index, 'walls']

        # Print how many roofs and walls were retrofitted
        # print(
        #     f"Retrofitting {len(vulnerable_houses)} houses: {len(vulnerable_houses[retrofitted_roofs != current_roofs])} roofs and {len(vulnerable_houses[retrofitted_walls != current_walls])} walls")

        households.loc[vulnerable_houses.index, 'retrofitted'] = True

        # Calculate the new vulnerability score
        households = recalculate_house_vulnerability(
            country, households, disaster_type)

        # NOTE: After retrofitting we do not randomize the vulnerability score
        return households

    else:
        raise ValueError(f"Country '{country}' not yet supported")


def recalculate_house_vulnerability(country: str, households: pd.DataFrame, disaster_type: str) -> pd.DataFrame:
    '''Recalculate vulnerability of a house based on its new retrofitted roof and walls material.

    Args:
        country (str): Country name.
        households (pd.DataFrame): Households.
            Required columns: 'roof', 'walls'
        disaster_type (str): Type of disaster. Example: "hurricane".

    Returns:
        pd.DataFrame: Households with recalculated vulnerability scores.

    Raises:
        ValueError: If the country is not supported.
        ValueError: If the disaster type is not supported.
    '''
    # Each country has different roof and wall materials and their respective vulnerability scores
    if country == 'Dominica':
        # Define vulnerability scores in dictionaries for easy mapping
        v_roof_scores = {
            'Concrete': 0.2,
            'Sheet metal (galvanize, galvalume)': 0.4,
            'Shingle (wood)': 0.6,
            'Shingle (asphalt)': 0.6,
            'Shingle (other)': 0.6,
            'Other': 0.75,
        }

        v_walls_scores = {
            'Brick/Blocks': 0.2,
            'Concrete/Concrete Blocks': 0.2,
            'Concrete/Concrete blocks': 0.2,
            'Wood & Concrete': 0.4,
            'Wood/Timber': 0.6,
            'Plywood': 0.7,
            'Makeshift': 0.8,
            "Other/Don't know": 0.8,
            "Other/Don't Know": 0.8,
            'Other': 0.8,
        }

    elif country == 'Nigeria':
        v_roof_scores = {
            'Finished – Concrete': 0.2,
            'Finished – Asbestos': 0.25,
            'Finished – Metal tile': 0.35,
            'Finished – Tile': 0.5,
            'Other – Specific': 0.75,
            'Rudimentary – Other': 0.75,
            'Other': 0.75,
            'Natural – Thatch/palm leaf': 0.9
        }

        v_walls_scores = {
            'Finished – Cement blocks': 0.2,
            'Finished – Stone with lime/cement': 0.4,
            'Finished – Woven Bamboo': 0.6,
            'Rudimentary – Bamboo with mud': 0.8,
            'Other': 0.8,
        }

    else:
        raise ValueError(f"Country '{country}' not yet supported")

    # Save current vulnerability score
    # current_v = households['v'].copy()
    # current_v_roof = households['v_roof'].copy()
    # current_v_walls = households['v_walls'].copy()

    # Calculate the vulnerability scores for the roof and walls
    households['v_roof'] = households['roof'].map(v_roof_scores)
    households['v_walls'] = households['walls'].map(v_walls_scores)

    # Count how many v values changed
    # print(
    #     f'Changed {len(households[households["v_roof"] != current_v_roof])} roofs')
    # print(
    #     f'Changed {len(households[households["v_walls"] != current_v_walls])} walls')

    # Calculate the new vulnerability score
    if disaster_type == 'hurricane':
        v_roof_multiplier = 0.6
        v_walls_multiplier = 0.4
    elif disaster_type == 'earthquake':
        v_roof_multiplier = 0.3
        v_walls_multiplier = 0.7
    elif disaster_type == 'flood':
        v_roof_multiplier = 0.2
        v_walls_multiplier = 0.8
    else:
        raise ValueError(f"Disaster type '{disaster_type}' not yet supported")

    # NOTE: Since it is for a retrofitting policy,
    # we immediately assign the new values to `v` and not `v_init` column
    households['v'] = v_roof_multiplier * \
        households['v_roof'] + v_walls_multiplier * households['v_walls']

    # Assertions to check values are within expected range
    assert households[['v_roof', 'v_walls', 'v']].apply(
        lambda x: (x >= 0).all() and (x <= 1).all()).all()
    assert households['v'].isna().sum() == 0

    return households


def apply_policy(households: pd.DataFrame, country: str, current_policy: str, disaster_type: str, random_seed: int) -> pd.DataFrame:
    '''
    Apply a specific policy to a set of households.

    This function applies a specific policy to a set of households, based on their economic status and the impact of external factors. The policy is specified in the format "<policy_type>:<policy_details>", where <policy_type> can be "asp" (cash transfer) or "retrofit" (housing retrofitting), and <policy_details> is a string containing the details of the policy.

    Args:
        country (str): Country name.
        households (pd.DataFrame): Households.
            Required columns: 'is_affected', 'is_poor', 'exp', 'povline_adjusted', 'keff', 'v', 'sav', 'roof', 'walls'
        policy (str): Policy to apply. Format: "<policy_type>:<policy_details>". Example: "asp:poor+100".
        disaster_type (str): Type of disaster. Example: "hurricane".
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
        return retrofitting(country, households, current_policy.split(':')[1], disaster_type, random_seed)
    elif policy_type == 'none':
        return households
    else:
        raise ValueError(
            f"Unknown policy type: {policy_type}, please use 'asp', 'retrofit' or 'none'")
