import numpy as np
import pandas as pd


def randomize(households: pd.DataFrame, risk_and_damage: pd.DataFrame,
              params: dict, random_seed: int, print_statistics: bool = False) -> pd.DataFrame:
    '''
    Randomizes household data and matches it with risk data.

    Args:
        households: Households.
        params: Dictionary with parameters for randomization.
        random_seed: Random seed.
        print_statistics: Whether to print statistics.

    Returns:
        households: Randomized households.
    '''
    # Ensure reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        households['random_seed'] = random_seed

    # Save country and return period
    households['country'] = params['country']
    households['return_period'] = params['return_period']

    # Extra necessary parameters
    rnd_inc_params = params['rnd_inc_params']
    rnd_sav_params = params['rnd_sav_params']
    rnd_rent_params = params['rnd_rent_params']
    avg_prod = params['avg_prod']
    rnd_house_vuln_params = params['rnd_house_vuln_params']
    min_households = params['min_households']
    atol = params['atol']

    # Step 1: Randomize income
    households = rnd_income(households, rnd_inc_params)

    # Step 2: Randomize savings
    households = rnd_savings(households, rnd_sav_params)

    # Step 3: Randomize rent
    households = rnd_rent(households, rnd_rent_params)

    # Step 4: Calculate dwelling value
    households = calc_dwelling_value(households, avg_prod)

    # Step 5: Randomize house vulnerability
    households = rnd_house_vuln(households, rnd_house_vuln_params)

    # Step 6: Resample households to ensure representativeness
    households = resample_households(households, min_households, random_seed)

    # Step 7: Match the survey with risk data
    households = match_survey_with_risk_data(
        households, risk_and_damage, atol, print_statistics)

    return households


# ---------------------------------------------------------------------------- #
#                        Randomize household survey data                       #
# ---------------------------------------------------------------------------- #


def rnd_inc_mult(size: int, **params) -> np.ndarray:
    # NOTE: `normal` is to showcase the flexibility of the function,
    # we cannot use it for income generation as it can produce negative values
    '''Randomize income multiplier based on the specified parameters.'''
    distr = params.get('distr')
    if distr == 'uniform':
        low = params.get('inc_mult') - params.get('delta')
        high = params.get('inc_mult') + params.get('delta')
        if low < 0:
            raise ValueError("Low income cannot be negative.")
        return np.random.uniform(low, high, size)
    elif distr == 'normal':
        loc = params.get('loc')
        scale = params.get('scale')
        return np.random.normal(loc, scale, size)
    else:
        raise ValueError(
            f"Distribution '{distr}' is not supported yet.")


def rnd_income(households: pd.DataFrame, rnd_income_params: dict) -> pd.DataFrame:
    '''Randomize household income.'''
    households = households.copy()
    size = households.shape[0]

    # Apply income generation for regions or country as applicable
    if households['rgn_inc_mult'].notna().any():
        for region in households['region'].unique():
            mask = households['region'] == region
            rnd_income_params['inc_mult'] = households.loc[mask,
                                                           'rgn_inc_mult'].iloc[0]
            size = mask.sum()
            inc_mult_rnd = rnd_inc_mult(size, **rnd_income_params)
            # Estimate income as a product of expenditure and income multiplier
            households.loc[mask, 'inc'] = households.loc[mask,
                                                         'exp'] * inc_mult_rnd
    elif households['ctry_inc_mult'].notna().any():
        rnd_income_params['inc_mult'] = households['ctry_inc_mult'].iloc[0]
        inc_mult_rnd = rnd_inc_mult(size, **rnd_income_params)
        # Estimate income as a product of expenditure and income multiplier
        households['inc'] = households['exp'] * inc_mult_rnd
    else:
        raise ValueError("No income multiplier found.")

    assert not households['inc'].isna().any(), "Income cannot be NaN"
    assert not (households['inc'] < 0).any(), "Income cannot be negative"
    assert not (households['inc'] == 0).any(), "Income cannot be 0"

    return households


def rnd_sav_mult(size: int, **params):
    """Randomize savings multiplier based on the specified distribution."""
    # NOTE: `normal` is to showcase the flexibility of the function,
    # we cannot use it for income generation as it can produce negative values
    distr = params.get('distr')
    if distr == 'uniform':
        low = params.get('low')
        high = params.get('high')
        return np.random.uniform(low, high, size)
    elif distr == 'normal':
        loc = params.get('loc')
        scale = params.get('scale')
        return np.random.normal(loc, scale, size)
    else:
        raise ValueError(f"Distribution '{distr}' is not supported yet.")


def rnd_savings(households: pd.DataFrame, rnd_savings_params: dict) -> pd.DataFrame:
    '''Randomize household savings.'''
    # Get function parameters
    distr = rnd_savings_params['distr']
    size = households.shape[0]

    # Ensure that the distribution is supported
    if distr == 'uniform':
        low = rnd_savings_params.get('avg') - rnd_savings_params.get('delta')
        high = rnd_savings_params.get('avg') + rnd_savings_params.get('delta')
        if low < 0:
            raise ValueError("Low savings cannot be negative.")
        rnd_savings_params.update({'low': low, 'high': high})

    else:
        raise ValueError(f"Distribution '{distr}' is not supported yet.")

    # Randomize savings multiplier using the distribution-specific function
    sav_mult_rnd = rnd_sav_mult(size, **rnd_savings_params)

    # Estimate savings as a product of income and savings multiplier
    households['sav'] = households['inc'] * sav_mult_rnd

    assert not households['sav'].isna().any(), "Savings cannot be NaN"
    assert not (households['sav'] < 0).any(), "Savings cannot be negative"
    assert not (households['sav'] == 0).any(), "Savings cannot be 0"

    return households


def rnd_rent_mult(size: int, **params):
    """Randomize rent multiplier based on the specified distribution."""
    # NOTE: `normal` is to showcase the flexibility of the function,
    # we cannot use it for income generation as it can produce negative values
    distr = params.get('distr')
    if distr == 'uniform':
        low = params.get('low')
        high = params.get('high')
        return np.random.uniform(low, high, size)
    elif distr == 'normal':
        loc = params.get('loc')
        scale = params.get('scale')
        return np.random.normal(loc, scale, size)
    else:
        raise ValueError(f"Distribution '{distr}' is not supported yet.")


def rnd_rent(households: pd.DataFrame, rnd_rent_params: dict) -> pd.DataFrame:
    '''Randomize household rent.'''
    # Get function parameters
    distr = rnd_rent_params['distr']
    size = households.shape[0]

    # Ensure that the distribution is supported
    if distr == 'uniform':
        low = rnd_rent_params.get('avg') - rnd_rent_params.get('delta')
        high = rnd_rent_params.get('avg') + rnd_rent_params.get('delta')
        if low < 0:
            raise ValueError("Low rent cannot be negative.")
        rnd_rent_params.update({'low': low, 'high': high})

    else:
        raise ValueError(f"Distribution '{distr}' is not supported yet.")

    # Generate savings using the distribution-specific function
    rent_mult_rnd = rnd_rent_mult(size, **rnd_rent_params)

    # Assign rent as a product of exp and rent multiplier to each of the rows
    households['exp_house'] = households['exp'].mul(rent_mult_rnd)

    assert not households['exp_house'].isna().any(), "Rent cannot be NaN"
    assert not (households['exp_house'] < 0).any(), "Rent cannot be negative"
    assert not (households['exp_house'] == 0).any(), "Rent cannot be 0"

    # Remove rent for the households that own
    households.loc[households['own_rent'] == 'own', 'exp_house'] = 0

    return households


def calc_dwelling_value(households: pd.DataFrame, avg_prod: float) -> pd.DataFrame:
    '''Calculate dwelling value based on the average productivity.'''
    # Get the dwelling value for the households that own
    households.loc[households['own_rent'] == 'own',
                   'k_house'] = households['inc'] / avg_prod

    # Set the dwelling value to 0 for the households that rent
    households.loc[households['own_rent'] == 'rent',
                   'k_house'] = 0

    assert not households['k_house'].isna(
    ).any(), "Dwelling value cannot be NaN"
    assert not (households['k_house'] < 0).any(
    ), "Dwelling value cannot be negative"

    return households


def rnd_house_vuln(households: pd.DataFrame, rnd_house_vuln_params: dict) -> pd.DataFrame:
    '''
    Randomize house vulnerability.

    Args:
        households: Households
            Required columns: 'v_init' (initial vulnerability)
        rnd_house_vuln_params: Dictionary with parameters for randomization.

    Returns:
        households: Households with randomized vulnerability.

    Raises:
        ValueError: If the distribution is not supported.
    '''
    distr = rnd_house_vuln_params['distr']
    if distr == 'uniform':
        # Unpack parameters
        low = rnd_house_vuln_params['low']  # default 0.8
        high = rnd_house_vuln_params['high']  # default 1.0
        thresh = rnd_house_vuln_params['thresh']  # default 0.9

        # Multiply initial vulnerability by a random value
        households['v'] = households['v_init'] * \
            np.random.uniform(low, high, households.shape[0])

        # Limit vulnerability to a threshold
        households.loc[households['v'] > thresh, 'v'] = thresh
    else:
        raise ValueError("Only uniform distribution is supported yet.")
    return households

# ---------------------------------------------------------------------------- #
#                         Duplicate households-related                         #
# ---------------------------------------------------------------------------- #


def resample_households(households: pd.DataFrame, min_households: int, random_seed: int) -> pd.DataFrame:
    '''Resample country households to be more representative in `identify_affected` function.

    Args:
        households (pd.DataFrame): Households.
        min_households (int): Minimum number of households for the country sample to be representative.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Resampled households.f
    '''
    sample = pd.DataFrame()
    households['id_orig'] = households['id']
    for rgn in households['region'].unique():
        rgn_households = households[households['region'] == rgn]
        # TODO: Remove this condition when the data is fixed
        if rgn != 'Borno':
            rgn_households = resample_region(
                rgn_households, min_households, random_seed)
        sample = pd.concat(
            [sample, rgn_households])
    sample['id'] = range(1, len(sample) + 1)
    sample.reset_index(drop=True, inplace=True)
    return sample


def resample_region(households: pd.DataFrame, min_households: int, random_seed: int) -> pd.DataFrame:
    '''Weighted resampling with adjustment for household representation within a given region.

    Args:
        households (pd.DataFrame): Households of a specific region.
        min_households (int): Minimum number of households for the region sample to be representative.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Resampled households of a specific region.

    Raises:
        ValueError: If the total weights after resampling is not equal to the initial total weights.
    '''

    if len(households) < min_households:
        # Save the initial total weights
        initial_total_weights = households['wgt'].sum()

        # Save the original household id
        households = households.assign(
            id_orig=households['id'])

        # Sample households with replacement
        delta = min_households - len(households)

        # To avoid sampling one household
        if delta == 1:
            delta = 2

        # ?: Check if fixing np random seed is affecting pandas random sampling
        if random_seed is not None:
            sample = households.sample(
                n=delta, replace=True, random_state=random_seed)
        else:
            sample = households.sample(
                n=delta, replace=True)

        # Keep how many duplicates by index
        duplicates = sample.index.value_counts()

        # Combine the original and the sampled households
        duplicated_households = pd.concat(
            [households, sample], axis=0, sort=False)

        # Iterate over the duplicated households and update the weights
        for household_id in duplicates.index:
            # Get original weight
            original_weight = households.loc[household_id, 'wgt']

            # Count how many rows are there (could be one as well)
            n_duplicates = duplicates[household_id]

            # Update the weight
            duplicated_households.loc[household_id,
                                      'wgt'] = original_weight / (n_duplicates + 1)

            # Get the new weight
            weights_after_duplication = duplicated_households.loc[household_id, 'wgt'].sum(
            )

            # Check if it is close to the original weight
            if not np.isclose(weights_after_duplication, original_weight, atol=1e-6):
                raise ValueError(
                    'Total weights after duplication is not equal to the initial total weights')

        if not np.isclose(duplicated_households['wgt'].sum(), initial_total_weights, atol=1e-6):
            raise ValueError(
                'Total weights after duplication is not equal to the initial total weights')

        return duplicated_households.reset_index(drop=True)
    else:
        return households

# ---------------------------------------------------------------------------- #
#       Matching household survey assets with risk and asset damage data       #
# ---------------------------------------------------------------------------- #


def match_survey_with_risk_data(households: pd.DataFrame, risk_and_damage: pd.DataFrame, atol: int, print_statistics: bool = False) -> pd.DataFrame:
    # Initialize list to store matched households
    matched = []

    # Iterate over unique regions
    for region in households['region'].unique():
        # Filter households by region
        region_households = households[households['region'] == region].copy()

        # Compute survey assets
        region_households['survey_assets'] = region_households[[
            'k_house', 'wgt']].prod(axis=1)
        survey_assets = region_households['survey_assets'].sum()

        # Get total exposed asset from disaster risk assessment for the given region
        disaster_assessment = risk_and_damage[risk_and_damage['region']
                                              == region].iloc[0]
        exposed_assets = disaster_assessment['total_exposed_stock']

        # Optionally print statistics
        if print_statistics:
            print_statistics_for_region(
                region, survey_assets, exposed_assets, region_households)

        # Match households with disaster risk assessment data
        df = match_assets(
            region_households, exposed_assets, atol=atol)

        # Recalculate survey assets
        df['survey_assets'] = df[['k_house', 'wgt']].prod(axis=1)

        # Assert that the assets match after matching
        assert round(exposed_assets) == round(
            df['survey_assets'].sum()), "Mismatch in assets after matching"

        matched.append(df)

    # Concatenate matched households and return
    return pd.concat(matched)


def print_statistics_for_region(region: str, survey_assets: float, exposed_assets: float, households: pd.DataFrame):
    print('Region:', region)
    print('Household survey assets:', '{:,}'.format(round(survey_assets)))
    print('Disaster risk assessment assets:',
          '{:,}'.format(round(exposed_assets)))
    comparison_ratio = survey_assets / \
        exposed_assets if survey_assets > exposed_assets else exposed_assets / survey_assets
    print('Comparison:', '{:.2f} times'.format(comparison_ratio), 'bigger')
    print('Median expenditure:', '{:,}'.format(households['exp'].median()))
    print('Poverty line:', '{:,}'.format(households['povline'].values[0]))
    print('----------------------------------------\n')


def check_asset_match(households: pd.DataFrame, tot_exposed_asset: float, atol: float) -> bool:
    '''Check if the total asset in the survey matches the total exposed asset.

    Args:
        households (pd.DataFrame): Households data.
        tot_exposed_asset (float): Total exposed asset stock from the damage data.
        atol (float): Absolute tolerance for the comparison.

    Returns:
        bool: True if the assets match within the specified tolerance, False otherwise.
    '''
    tot_asset_surv = households[['k_house', 'wgt']].prod(axis=1).sum()
    return np.isclose(tot_exposed_asset, tot_asset_surv, atol=atol)


def adjust_assets(households: pd.DataFrame, tot_exposed_asset: float) -> pd.DataFrame:
    '''Adjusts the assets of the households to match the total exposed asset.

    Args:
        households (pd.DataFrame): Households data.
        tot_exposed_asset (float): Total exposed asset to match.

    Returns:
        pd.DataFrame: Adjusted households data.
    '''
    tot_asset_surv = households[['k_house', 'wgt']].prod(axis=1).sum()
    scaling_factor = tot_exposed_asset / tot_asset_surv
    households['tot_asset_surv'] = tot_asset_surv
    households['k_house'] *= scaling_factor

    return households


def adjust_expenditure(households: pd.DataFrame, scaling_factor: float) -> pd.DataFrame:
    '''Adjusts the expenditure of the households based on the scaling factor.

    Args:
        households (pd.DataFrame): Households data.
        scaling_factor (float): Scaling factor for adjustment.

    Returns:
        pd.DataFrame: Households data with adjusted expenditure.
    '''
    households['exp'] *= scaling_factor
    households['exp_house'] *= scaling_factor

    return households


def adjust_poverty_line(households: pd.DataFrame, scaling_factor: float) -> pd.DataFrame:
    '''Adjusts the poverty line of the households based on the scaling factor.

    Args:
        households (pd.DataFrame): Households data.
        scaling_factor (float): Scaling factor for adjustment.

    Returns:
        pd.DataFrame: Households data with adjusted poverty line.
    '''
    poverty_line = households['poverty_line'].iloc[0]
    households['poverty_line_adjusted'] = poverty_line * scaling_factor

    return households


def match_assets(households: pd.DataFrame, tot_exposed_asset: float, atol: float) -> pd.DataFrame:
    '''Matches and possibly adjusts assets, poverty line and expenditure of households to the asset damage data.

    Args:
        households (pd.DataFrame): Households data.
        tot_exposed_asset (float): Total exposed asset from the damage data.
        atol (float): Absolute tolerance for the comparison.

    Returns:
        pd.DataFrame: Households with potentially adjusted assets, expenditure, and poverty line.
    '''
    if not check_asset_match(households, tot_exposed_asset, atol):
        tot_asset_surv = households[['k_house', 'wgt']].prod(axis=1).sum()
        scaling_factor = tot_exposed_asset / tot_asset_surv
        households = adjust_assets(households, tot_exposed_asset)
        households = adjust_expenditure(households, scaling_factor)
        households = adjust_poverty_line(households, scaling_factor)

    return households
