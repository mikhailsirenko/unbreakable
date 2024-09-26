import pandas as pd
from typing import Dict


def calculate_socioeconomic_resilience(
    affected_households: pd.DataFrame,
) -> Dict[str, float]:
    """Calculate socio-economic resilience of affected households.

    Socio-economic resilience is calculated using two methods:
    1. Consumption-based = Asset damage / consumption loss
    2. Wellbeing-based = Asset damage / consumption equivalent loss

    Args:
        affected_households (pd.DataFrame): Affected households.

    Returns:
        Dict[str, float]: Dictionary containing:
            - consumption_based: Resilience based on consumption loss
            - equivalent_based: Resilience based on consumption equivalent loss
    """
    total_asset_damage = (
        affected_households[["keff", "v", "household_weight"]].prod(axis=1)
    ).sum()

    total_consumption_loss = (
        affected_households[["consumption_loss_npv", "household_weight"]].prod(axis=1)
    ).sum()

    total_consumption_equivalent_loss = calculate_consumption_equivalent_loss(
        affected_households
    )

    if total_consumption_loss == 0 or total_consumption_equivalent_loss == 0:
        return {
            "consumption_based": 0,
            "wellbeing_based": 0,
        }

    else:
        return {
            "consumption_based": total_asset_damage / total_consumption_loss,
            "wellbeing_based": total_asset_damage / total_consumption_equivalent_loss,
        }


def calculate_consumption_equivalent_loss(affected_households: pd.DataFrame) -> float:
    """
    Calculate the total consumption equivalent loss for affected households.

    This function computes the sum of weighted wellbeing losses relative to welfare
    across all affected households.

    Args:
        affected_households (pd.DataFrame): DataFrame containing affected households data.

    Returns:
        float: The total consumption equivalent loss, always returned as a positive number.

    Note:
        The function assumes that 'wellbeing_loss' is typically a negative value,
        hence the negative sign in the calculation to return a positive loss.
    """
    total_consumption_equivalent_loss = -(
        (affected_households["wellbeing_loss"] / affected_households["welfare"])
        * affected_households["household_weight"]
    ).sum()

    return total_consumption_equivalent_loss
