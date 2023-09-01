import pandas as pd
import numpy as np
from unbreakable.modules.households import *


def calculate_wprime(all_households: pd.DataFrame, all_damage: pd.DataFrame, districts: list, return_period: int, min_representative_households: int, random_seed: int, poverty_line: float, indigence_line: float, atol: float, consump_util: float) -> float:
    '''Calculate `wprime` for the whole country.

    `wprime` is a factor that converts an abstract concept of wellbeing loss 
    into consumption loss in monetary terms.

    Args:
        all_households (pd.DataFrame): Households from all districts.
        all_damage (pd.DataFrame): Damage data for all districts.
        districts (list): A list of districts.
        return_period (int): Return period.
        min_representative_households (int): Minimum number of households to be representative.
        random_seed (int): Random seed.
        poverty_line (float): Poverty line.
        indigence_line (float): Indigence line.
        atol (float): Abs tolerance for matching asset stock of damage and household survey data sets.
        consump_util (float): Consumption utility.

    Returns:
        float: `wprime` for the whole country.
    '''

    households_adjusted = []
    for district in districts:
        tot_exposed_asset = all_damage.loc[(all_damage['district'] == district) & (
            all_damage['rp'] == return_period), 'exposed_value'].values[0]

        households = all_households[all_households['district'] == district].copy(
        )
        households = (households.pipe(duplicate_households, min_representative_households, random_seed)
                                .pipe(match_assets_and_expenditure, tot_exposed_asset, poverty_line, indigence_line, atol))
        households_adjusted.append(households)

    households_adjusted = pd.concat(households_adjusted)
    wprime = (np.sum(
        households_adjusted['aeexp'] * households_adjusted['popwgt']) / np.sum(households_adjusted['popwgt']))**(-consump_util)
    return wprime
