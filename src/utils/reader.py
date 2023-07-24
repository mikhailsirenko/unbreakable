import pandas as pd
import os
import numpy as np
import json


def read_asset_damage(country) -> None:
    '''Read asset damage for all districts from a XLSX file and load it into the memory.'''
    if country == 'Saint Lucia':
        all_damage = pd.read_excel(
            f"../data/processed/asset_damage/{country}/{country}.xlsx", index_col=None, header=0)
    else:
        raise ValueError('Only `Saint Lucia` is supported.')

    return all_damage


def get_asset_damage(all_damage: pd.DataFrame, scale: str, district: str, return_period: int, print_statistics: bool) -> None:
    '''Get asset damage for a specific district.'''
    if scale == 'district':
        event_damage = all_damage.loc[(all_damage[scale] == district) & (
            all_damage['rp'] == return_period), 'pml'].values[0]  # PML
        total_asset_stock = all_damage.loc[(all_damage[scale] == district) & (
            all_damage['rp'] == return_period), 'exposed_value'].values[0]  # Exposed value

    else:
        raise ValueError(
            'Only `district` scale is supported.')

    event_damage = event_damage
    total_asset_stock = total_asset_stock
    expected_loss_fraction = event_damage / total_asset_stock

    if expected_loss_fraction > 1:
        raise ValueError(
            'Expected loss fraction is greater than 1. Check the data.')

    if print_statistics:
        print('Event damage = ' + str('{:,}'.format(round(event_damage))))
        print('Total asset stock = ' +
              str('{:,}'.format(round(total_asset_stock))))
        print('Expected loss fraction = ' +
              str(np.round(expected_loss_fraction, 3)))

    return event_damage, total_asset_stock, expected_loss_fraction


def read_household_survey(country: str) -> pd.DataFrame:
    '''Reads household survey from a CSV file.

    Args:
        country (str): Country name.

    Returns:
        pd.DataFrame: Household survey data.

    '''
    household_survey = pd.read_csv(
        f"../data/processed/household_survey/{country}/{country}.csv")
    return household_survey


def duplicate_households(household_survey: pd.DataFrame, min_households: int) -> pd.DataFrame:
    '''Duplicates households if the number of households is less than `min_households` threshold.

    Args:
        household_survey (pd.DataFrame): Household survey data.
        min_households (int): Minimum number of households.

    Returns:
        pd.DataFrame: Household survey data with duplicated households.

    Raises:
        ValueError: If the total weights after duplication is not equal to the initial total weights.
    '''

    if len(household_survey) < min_households:
        print(
            f'Number of households = {len(household_survey)} is less than the threshold = {min_households}')

        initial_total_weights = household_survey['popwgt'].sum()

        # Save the original household id
        household_survey['hhid_original'] = household_survey[[
            'hhid']]

        # Get random ids from the household data to be duplicated
        ids = np.random.choice(
            household_survey.index, min_households - len(household_survey), replace=True)
        n_duplicates = pd.Series(ids).value_counts() + 1
        duplicates = household_survey.loc[ids]

        # Adjust the weights of the duplicated households
        duplicates['popwgt'] = duplicates['popwgt'] / n_duplicates

        # Adjust the weights of the original households
        household_survey.loc[ids, 'popwgt'] = household_survey.loc[ids, 'popwgt'] / \
            n_duplicates

        # Combine the original and duplicated households
        household_survey = pd.concat(
            [household_survey, duplicates], ignore_index=True)

        # Check if the total weights after duplication is equal to the initial total weights
        # TODO: Allow for a small difference
        weights_after_duplication = household_survey['popwgt'].sum()
        if weights_after_duplication != initial_total_weights:
            raise ValueError(
                'Total weights after duplication is not equal to the initial total weights')

        household_survey.reset_index(drop=True, inplace=True)
        print(
            f'Number of households after duplication: {len(household_survey)}')
    else:
        return household_survey


def calculate_average_productivity(households: pd.DataFrame, print_statistics: bool) -> float:
    '''Calculate average productivity as aeinc \ k_house_ae.

    Args:
        print_statistics (bool, optional): Whether to print the average productivity. Defaults to False.

    Returns:
        float: Average productivity.
    '''
    # DEFF: aeinc - some type of income
    average_productivity = households['aeinc'] / \
        households['k_house_ae']

    # ?: What's happening here?
    # average_productivity = average_productivity.iloc[0]
    average_productivity = np.nanmedian(average_productivity)
    if print_statistics:
        print('Average productivity of capital = ' +
              str(np.round(average_productivity, 3)))
    average_productivity = average_productivity
    return average_productivity


def adjust_assets_and_expenditure(households: pd.DataFrame, total_asset_stock: float, poverty_line: float, indigence_line: float, print_statistics: bool) -> None:
    '''Adjust assets and expenditure of household to match data of asset damage file.

    There can be a mismatch between the data in the household survey and the of the asset damage.
    The latest was created independently.'''

    # ?: Do we always have to do that?
    # If yes, remove the corresponding variable. Or else add a condition?

    # k_house_ae - effective capital stock of the household
    # aeexp - adult equivalent expenditure of a household (total)
    # aeexp_house - data['hhexp_house'] (annual rent) / data['hhsize_ae']
    included_variables = ['k_house_ae', 'aeexp', 'aeexp_house']

    # Save the initial values
    households['k_house_ae_original'] = households['k_house_ae']
    households['aeexp_original'] = households['aeexp']
    households['aeexp_house_original'] = households['aeexp_house']

    total_asset_in_survey = households[[
        'popwgt', 'k_house_ae']].prod(axis=1).sum()
    households['total_asset_in_survey'] = total_asset_in_survey
    total_asset_in_survey = total_asset_in_survey
    scaling_factor = total_asset_stock / total_asset_in_survey
    households[included_variables] *= scaling_factor
    households['poverty_line_adjusted'] = poverty_line * \
        scaling_factor
    households['indigence_line_adjusted'] = indigence_line * \
        scaling_factor

    if print_statistics:
        print('Total asset in survey =',
              '{:,}'.format(round(total_asset_in_survey)))
        print('Total asset in asset damage file =',
              '{:,}'.format(round(total_asset_stock)))
        print('Scaling factor =', round(scaling_factor, 3))

    return households


def calculate_pml(households: pd.DataFrame, expected_loss_fraction: float, print_statistics: bool) -> None:
    '''Calculate probable maxmium loss as a product of population weight, effective capital stock and expected loss fraction.'''
    # DEF: keff - effective capital stock
    # DEF: pml - probable maximum loss
    # DEF: popwgt - population weight of each household
    households['keff'] = households['k_house_ae'].copy()
    pml = households[['popwgt', 'keff']].prod(
        axis=1).sum() * expected_loss_fraction

    households['pml'] = pml
    if print_statistics:
        print('Probable maximum loss (total) : ',
              '{:,}'.format(round(pml)))
    return households
