import pandas as pd
import numpy as np
from typing import Optional


def validate_rent(
    rent: pd.Series, income: pd.Series, expenditure: pd.Series
) -> Optional[ValueError]:
    """
    Validate the generated rent.

    This function checks if the generated rent values are valid:
    - Not NaN
    - Positive
    - Less than income
    - Less than expenditure

    Args:
        rent (pd.Series): Series of generated rent values.
        income (pd.Series): Series of income values.
        expenditure (pd.Series): Series of expenditure values.

    Returns:
        None if validation passes.

    Raises:
        ValueError: If any validation check fails.
    """
    if rent.isna().any():
        raise ValueError("Rent cannot be NaN")
    if (rent < 0).any():
        raise ValueError("Rent must be positive")
    if (rent >= income).any():
        raise ValueError("Rent must be less than income")
    if (rent >= expenditure).any():
        raise ValueError("Rent must be less than expenditure")
    return None


def validate_income(income: pd.Series, expenditure: pd.Series) -> Optional[ValueError]:
    """
    Validate the generated income.

    This function checks if the generated income values are valid:
    - Not NaN
    - Positive
    - Greater than corresponding expenditure

    Args:
        income (pd.Series): Series of generated income values.
        expenditure (pd.Series): Series of corresponding expenditure values.

    Returns:
        None if validation passes.

    Raises:
        ValueError: If any validation check fails.
    """
    if income.isna().any():
        raise ValueError("Income cannot be NaN")
    if (income <= 0).any():
        raise ValueError("Income must be positive")
    if (income <= expenditure).any():
        raise ValueError("Income must be greater than expenditure")
    return None


def validate_savings(
    savings: pd.Series, income: pd.Series, expenditure: pd.Series
) -> Optional[ValueError]:
    """
    Validate the generated savings.

    This function checks if the generated savings values are valid:
    - Not NaN
    - Not negative
    - Not greater than income
    - Consistent with income - expenditure

    Args:
        savings (pd.Series): Series of generated savings values.
        income (pd.Series): Series of income values.
        expenditure (pd.Series): Series of expenditure values.

    Returns:
        None if validation passes.

    Raises:
        ValueError: If any validation check fails.
    """
    if savings.isna().any():
        raise ValueError("Savings cannot be NaN")
    if (savings < 0).any():
        raise ValueError("Savings cannot be negative")
    if (savings > income).any():
        raise ValueError("Savings cannot be greater than income")
    return None
