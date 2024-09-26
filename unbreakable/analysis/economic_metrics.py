import pandas as pd
from typing import Tuple


def calculate_average_annual_consumption_loss(
    affected_households: pd.DataFrame, max_years: int
) -> Tuple[float, float]:
    """
    Calculate the average annual consumption loss and its percentage of average annual consumption.

    Args:
        affected_households (pd.DataFrame): Affected households data. Must include columns:
            - consumption_loss_npv: Net present value of consumption loss
            - household_weight: Household weight
            - exp: Annual consumption (expenditure)
        max_years (int): Number of years cut-off parameter for calculating consumption loss.

    Returns:
        Tuple[float, float]:
            - Average annual consumption loss
            - Average annual consumption loss as a percentage of average annual consumption

    Raises:
        ValueError: If the average annual consumption loss percentage is greater than 100%
        ValueError: If the DataFrame is empty
    """
    if affected_households.empty:
        raise ValueError("The affected_households DataFrame is empty.")

    # Calculate annual consumption loss
    annual_consumption_loss = (
        affected_households["consumption_loss_npv"]
        .div(max_years)
        .mul(affected_households["household_weight"])
        .sum()
    )

    # Calculate weighted average annual consumption loss
    total_weight = affected_households["household_weight"].sum()
    annual_average_consumption_loss = annual_consumption_loss / total_weight

    # Calculate percentage of average annual consumption loss
    annual_average_consumption_loss_pct = (
        affected_households["consumption_loss_npv"]
        .div(max_years)
        .div(affected_households["exp"])
        .mul(affected_households["household_weight"])
        .sum()
    ) / total_weight

    if annual_average_consumption_loss_pct > 1:
        raise ValueError(
            "Annual average consumption loss percentage is greater than 100%"
        )

    return annual_average_consumption_loss, annual_average_consumption_loss_pct
