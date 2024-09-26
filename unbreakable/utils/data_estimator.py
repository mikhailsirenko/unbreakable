import pandas as pd
import numpy as np
from typing import Optional


def estimate_effective_capital_stock(
    households: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """
    Estimate effective capital stock for homeowners.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        params (dict): Dictionary of parameters for estimation.

    Returns:
        pd.DataFrame: DataFrame with estimated capital stock.

    Raises:
        ValueError: If 'inc' or 'owns_house' columns are not present in the DataFrame.
    """
    if "inc" not in households.columns or "owns_house" not in households.columns:
        raise ValueError(
            "Both 'inc' and 'owns_house' columns must be present to estimate capital stock."
        )

    households = households.copy()
    households.loc[households["owns_house"] == True, "k_house"] = (
        households["inc"] / params["economic_params"]["average_productivity"]
    )

    households.loc[households["owns_house"] == False, "k_house"] = 0

    # NOTE
    # Keff is the effective capital stock of the household
    # Physical assets such as land, homes, and durable goods used to generate income
    # For now we assume that keff is equal to the dwelling value k_house
    households["keff"] = households["k_house"]

    # TODO Estimate effective capital stock for renters in a more realistic way
    # Estimate effective capital stock for renters based on savings rate
    savings_rate = (households["inc"] - households["exp"]) / households["inc"]

    households.loc[households["owns_house"] == False, "keff"] = (
        households["inc"]
        * savings_rate
        / params["economic_params"]["average_productivity"]
    )

    return households


def estimate_welfare(households: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estimate welfare based on consumption utility function.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        params (dict): Dictionary of parameters for estimation.

    Returns:
        pd.DataFrame: DataFrame with estimated welfare.

    Raises:
        ValueError: If 'exp' column is not present in the DataFrame.
    """
    if "exp" not in households.columns:
        raise ValueError("'exp' column must be present to estimate welfare.")

    households = households.copy()

    # TODO Review the equation since it contradicts with Walsh & Hallegatte (2020)
    weighted_average_expenditure = np.sum(
        households["exp"] * households["household_weight"]
    ) / np.sum(households["household_weight"])
    welfare = weighted_average_expenditure ** (
        -params["economic_params"]["consumption_utility"]
    )
    households["welfare"] = welfare
    return households


def estimate_rent(households: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estimate household rent as percentage of income.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        params (dict): Dictionary of parameters for estimation.

    Returns:
        pd.DataFrame: DataFrame with estimated rent.

    Raises:
        ValueError: If 'inc' column is not present in the DataFrame.
    """
    if "inc" not in households.columns:
        raise ValueError("'inc' column must be present to estimate rent.")

    households = households.copy()
    pct_of_income = params.get("pct_of_income", 0.3)
    households["rent"] = households["inc"] * pct_of_income
    return households


def estimate_savings(households: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estimate household savings based on income and expenditure.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        params (dict): Dictionary of parameters for estimation.

    Returns:
        pd.DataFrame: DataFrame with estimated savings.

    Raises:
        ValueError: If 'inc' or 'exp' columns are not present in the DataFrame.
    """
    if "inc" not in households.columns or "exp" not in households.columns:
        raise ValueError(
            "Both 'inc' and 'exp' columns must be present to estimate savings."
        )

    households = households.copy()
    if params.get("cap_with_max_savings_rate", False):
        max_savings_rate = params.get("max_savings_rate", 0.05)
        households["sav"] = households["inc"] * max_savings_rate

    else:
        households["sav"] = households["inc"] - households["exp"]
    return households


def estimate_income(households: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estimate household incomes based on expenditure and income-to-expenditure ratios.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        params (dict): Dictionary of parameters for estimation.

    Returns:
        pd.DataFrame: DataFrame with estimated incomes.

    Raises:
        ValueError: If no income-to-expenditure ratio is found or provided.
    """
    households = households.copy()

    # If we previously inferred region-specific ratios, use them
    if "region_inc_exp_ratio" in households.columns:
        for region in households["region"].unique():
            mask = households["region"] == region
            region_ratio = households.loc[mask, "region_inc_exp_ratio"].iloc[0]
            households.loc[mask, "inc"] = households.loc[mask, "exp"] * region_ratio

    # If we manage to infer only country-wise ratio, use it
    elif "country_inc_exp_ratio" in households.columns:
        country_ratio = households["country_inc_exp_ratio"].iloc[0]
        households["inc"] = households["exp"] * country_ratio

    # If we don't have either, then use the ratio provided in config
    elif params.get("inc_exp_ratio"):
        inc_exp_ratio = params["inc_exp_ratio"]
        households["inc"] = households["exp"] * inc_exp_ratio

    else:
        raise ValueError(
            "No income-to-expenditure ratio found. Please provide region_inc_exp_ratio, country_inc_exp_ratio, or inc_exp_ratio in params."
        )

    return households
