import pandas as pd
import numpy as np


def align_household_assets_with_exposure_data(
    households: pd.DataFrame, exposure_data: pd.DataFrame, atol: float
) -> pd.DataFrame:
    """
    Aligns household assets with exposure data and scales household economic indicators
    for each spatial unit.

    This function performs the following steps for each unique spatial unit:
    1. Matches household data with corresponding exposure data.
    2. Aligns the total household assets with the exposed assets from the exposure data.
    3. If there's a mismatch (within the specified tolerance), it scales the household
       economic indicators to match the exposure data.

    Args:
        households (pd.DataFrame): Household survey data containing asset information
            and other economic indicators.
        exposure_data (pd.DataFrame): Exposure data containing information about total
            exposed assets for each spatial unit.
        atol (float): Absolute tolerance used for matching assets. If the difference
            between household assets and exposed assets is within this tolerance,
            no scaling is performed.

    Returns:
        pd.DataFrame: A DataFrame containing the aligned and potentially scaled
        household data. If scaling was performed, original values are preserved
        in columns with '_orig' suffix.

    Note:
        - The function assumes that both 'households' and 'exposure_data' DataFrames
          have a 'spatial_unit' column for matching.
        - The 'exposure_data' DataFrame should have a 'total_exposed_stock' column
          representing the total exposed assets for each spatial unit.
        - If scaling is performed, it affects the following columns in the household data:
          'exp', 'inc', 'sav', 'rent', 'k_house', and 'poverty_line'.
    """
    aligned_households = []
    for spatial_unit in households["spatial_unit"].unique():
        unit_households = households[households["spatial_unit"] == spatial_unit].copy()
        exposed_assets = exposure_data.loc[
            exposure_data["spatial_unit"] == spatial_unit, "total_exposed_stock"
        ].iloc[0]
        aligned_df = align_assets(unit_households, exposed_assets, atol)
        aligned_households.append(aligned_df)
    return pd.concat(aligned_households)


def align_assets(
    households: pd.DataFrame, exposed_assets: float, atol: float
) -> pd.DataFrame:
    """
    Align household assets with exposed assets from disaster risk data.

    Args:
        households (pd.DataFrame): Household data for a specific spatial unit.
        exposed_assets (float): Total exposed assets for the spatial unit.
        atol (float): Absolute tolerance for asset matching.

    Returns:
        pd.DataFrame: Households with aligned assets and adjusted values.
    """
    survey_assets = (households["keff"] * households["household_weight"]).sum()

    if not np.isclose(exposed_assets, survey_assets, atol=atol):
        scaling_factor = exposed_assets / survey_assets
        households = scale_household_data(households, scaling_factor)

    survey_assets_aligned = (households["keff"] * households["household_weight"]).sum()

    assert round(exposed_assets) == round(
        survey_assets_aligned
    ), "Asset mismatch after alignment"

    return households


def scale_household_data(
    households: pd.DataFrame, scaling_factor: float
) -> pd.DataFrame:
    """
    Scale household data based on the calculated scaling factor.

    Args:
        households (pd.DataFrame): Household data to be scaled.
        scaling_factor (float): Factor to scale the data by.

    Returns:
        pd.DataFrame: Scaled household data.
    """
    households["poverty_line_orig"] = households["poverty_line"].iloc[0]
    households["k_house_orig"] = households["k_house"]

    for column in ["exp", "inc", "sav", "rent", "k_house", "poverty_line"]:
        households[column] *= scaling_factor

    return households
