import pandas as pd
import numpy as np
from typing import Optional
from unbreakable.utils.data_validator import *
from unbreakable.utils.data_estimator import *
from unbreakable.utils.data_randomizer import *
from unbreakable.utils.data_resampler import *
from unbreakable.utils.data_matcher import *


def prepare_household_data(
    households: pd.DataFrame,
    disaster_impacts: pd.DataFrame,
    params: dict,
    random_seed: int,
) -> pd.DataFrame:
    """
    Prepare household data by updating, estimating and/or randomizing them based on provided parameters.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        disaster_impacts (pd.DataFrame): DataFrame containing disaster impacts data.
        params (dict): Dictionary of parameters for randomization.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with updated household data.
    """
    np.random.seed(random_seed)
    households = get_income(households, params["income_params"], random_seed)
    households = get_savings(households, params["savings_params"], random_seed)
    households = get_rent(households, params["rent_params"], random_seed)

    # Resample households to ensure minimum number of households for representative sample
    households = resample_households_by_spatial_unit(
        households, params["min_households"], random_seed
    )

    # Estimate the dwelling value
    households = estimate_dwelling_value(households, params["dwelling_params"])

    if params["disaster_params"]["impact_data_type"] == "assets":
        # There could be a mismatch between the assets in the household data and the exposure data
        households = align_household_assets_with_exposure_data(
            households, disaster_impacts, params["atol"]
        )

    # * This must be done after resampling and matching
    households = estimate_effective_capital_stock(
        households, params["effective_capital_stock_params"]
    )

    # Estimate welfare based on consumption utility
    households = estimate_welfare(households, params)

    return households


def get_rent(households: pd.DataFrame, params: dict, random_seed: int) -> pd.DataFrame:
    """
    Estimate and/or randomize household rent based on provided parameters.

    This function serves as a wrapper to call the estimate_rent and randomize_rent
    functions based on the provided parameters.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        params (dict): Dictionary of parameters for estimation and randomization.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with estimated and/or randomized rent.

    Raises:
        ValueError: If invalid combination of estimate and randomize parameters are provided.
    """
    estimate = params.get("estimate", False)
    randomize = params.get("randomize", False)

    if estimate and not randomize:
        households = estimate_rent(households, params)
    elif randomize and not estimate:
        if "rent" not in households.columns:
            raise ValueError(
                "'rent' column not found. Cannot randomize without existing rent estimates."
            )
        households = randomize_rent(households, params, random_seed)
    elif estimate and randomize:
        households = estimate_rent(households, params)
        households = randomize_rent(households, params, random_seed)
    else:
        # If neither estimate nor randomize is True, return the original households
        return households

    validate_rent(households["rent"], households["inc"], households["exp"])
    return households


def get_savings(
    households: pd.DataFrame, params: dict, random_seed: int
) -> pd.DataFrame:
    """
    Estimate and/or randomize household savings based on provided parameters.

    This function serves as a wrapper to call the estimate_savings and randomize_savings
    functions based on the provided parameters.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        params (dict): Dictionary of parameters for estimation and randomization.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with estimated and/or randomized savings.

    Raises:
        ValueError: If invalid combination of estimate and randomize parameters are provided.
    """
    estimate = params.get("estimate", False)
    randomize = params.get("randomize", False)

    if estimate and not randomize:
        households = estimate_savings(households, params)
    elif randomize and not estimate:
        if "sav" not in households.columns:
            raise ValueError(
                "'sav' column not found. Cannot randomize without existing savings estimates."
            )
        households = randomize_savings(households, params, random_seed)
    elif estimate and randomize:
        households = estimate_savings(households, params)
        households = randomize_savings(households, params, random_seed)
    else:
        # If neither estimate nor randomize is True, return the original households
        return households

    validate_savings(households["sav"], households["inc"], households["exp"])
    return households


def get_income(
    households: pd.DataFrame, params: dict, random_seed: int
) -> pd.DataFrame:
    """
    Estimate and/or randomize household incomes based on provided parameters.

    This function serves as a wrapper to call the estimate_income and randomize_income
    functions based on the provided parameters.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        params (dict): Dictionary of parameters for estimation and randomization.
        random_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with estimated and/or randomized incomes.

    Raises:
        ValueError: If neither estimate nor randomize parameters are provided.
    """
    estimate = params.get("estimate", False)
    randomize = params.get("randomize", False)

    if estimate and not randomize:
        households = estimate_income(households, params)

    elif randomize and not estimate:
        if "inc" not in households.columns:
            raise ValueError(
                "'inc' column not found. Cannot randomize without existing income estimates."
            )
        households = randomize_income(households, params, random_seed)

    elif estimate and randomize:
        households = estimate_income(households, params)
        households = randomize_income(households, params, random_seed)
    else:
        # If neither estimate nor randomize is True, return the original households
        return households

    validate_income(households["inc"], households["exp"])
    return households
