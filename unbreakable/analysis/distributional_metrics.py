import pandas as pd
import numpy as np
from typing import Dict, List, Any
from functools import partial
from unbreakable.analysis.economic_metrics import (
    calculate_average_annual_consumption_loss,
)
from unbreakable.analysis.resilience_metrics import calculate_socioeconomic_resilience


def calculate_distributional_impacts(
    households: pd.DataFrame,
    max_years: int,
    socioeconomic_attributes: List[str],
    baseline_avg_annual_consumption_loss_pct: float,
    baseline_resilience: Dict[str, float],
) -> Dict[str, float]:
    """
    Calculate distributional impacts across different socioeconomic groups.

    This function analyzes the impact of disasters on various socioeconomic groups
    by calculating metrics such as consumption loss and resilience for each group.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        max_years (int): Maximum number of years for calculating annualized consumption loss.
        socioeconomic_attributes (List[str]): List of column names representing socioeconomic attributes.
        baseline_avg_annual_consumption_loss_pct (float): Baseline average annual consumption loss percentage.
        baseline_resilience (Dict[str, float]): Baseline resilience metrics.

    Returns:
        Dict[str, float]: A dictionary containing calculated impact metrics for each socioeconomic group.
    """
    outcomes = {}
    # Filter for affected households
    affected_households = households[households["ever_affected"]]

    # Iterate through each socioeconomic attribute
    for column in socioeconomic_attributes:
        # Get unique, non-null values for the current attribute
        group_values = households[column].dropna().unique()

        # Calculate impacts for each group within the current attribute
        for group_value in group_values:
            # Filter affected households for the current group
            group_households = affected_households[
                affected_households[column] == group_value
            ]

            if group_households.empty:
                # Handle empty groups by passing empty DataFrame and default values
                group_outcomes = calculate_group_outcomes(
                    column,
                    group_value,
                    pd.DataFrame(),
                    affected_households,
                    0.0,
                    0.0,
                    {"consumption_based": 0.0, "wellbeing_based": 0.0},
                    baseline_avg_annual_consumption_loss_pct,
                    baseline_resilience,
                )
            else:
                # Calculate metrics for non-empty groups
                avg_consumption_loss, avg_consumption_loss_pct = (
                    calculate_average_annual_consumption_loss(
                        group_households, max_years
                    )
                )
                resilience = calculate_socioeconomic_resilience(group_households)

                group_outcomes = calculate_group_outcomes(
                    column,
                    group_value,
                    group_households,
                    affected_households,
                    avg_consumption_loss,
                    avg_consumption_loss_pct,
                    resilience,
                    baseline_avg_annual_consumption_loss_pct,
                    baseline_resilience,
                )

            outcomes.update(group_outcomes)

    return outcomes


def calculate_group_outcomes(
    column: str,
    group_value: Any,
    group_households: pd.DataFrame,
    affected_households: pd.DataFrame,
    average_consumption_loss: float,
    average_consumption_loss_pct: float,
    resilience: Dict[str, float],
    baseline_avg_annual_consumption_loss_pct: float,
    baseline_resilience: Dict[str, float],
) -> Dict[str, float]:
    """
    Calculate impact outcomes for a specific socioeconomic group.

    This function computes various impact metrics for a given socioeconomic group,
    including consumption loss, reconstruction rates, and resilience measures.

    Args:
        column (str): Name of the socioeconomic attribute.
        group_value (Any): The specific value of the socioeconomic attribute for this group.
        group_households (pd.DataFrame): DataFrame containing data for households in this group.
        affected_households (pd.DataFrame): DataFrame containing data for all affected households.
        average_consumption_loss (float): Average annual consumption loss for this group.
        average_consumption_loss_pct (float): Average consumption loss percentage for this group.
        resilience (Dict[str, float]): Resilience metrics for this group.
        baseline_avg_annual_consumption_loss_pct (float): Baseline average annual consumption loss percentage.
        baseline_resilience (Dict[str, float]): Baseline resilience metrics.

    Returns:
        Dict[str, float]: A dictionary containing calculated impact metrics for the specific group.
    """
    if group_households.empty:
        return {
            # f"annual_average_consumption_loss_{column}_{group_value}": None,
            f"annual_average_consumption_loss_pct_{column}_{group_value}": None,
            f"average_consumption_loss_pct_difference_{column}_{group_value}": None,
            # f"average_reconstruction_rate_{column}_{group_value}": None,
            f"average_reconstruction_rate_difference_{column}_{group_value}": None,
            # f"r_consumption_based_{column}_{group_value}": None,
            f"r_consumption_based_difference_{column}_{group_value}": None,
            # f"r_wellbeing_based_{column}_{group_value}": None,
            f"r_wellbeing_based_difference_{column}_{group_value}": None,
        }

    return {
        # f"annual_average_consumption_loss_{column}_{group_value}": average_consumption_loss,
        f"annual_average_consumption_loss_pct_{column}_{group_value}": average_consumption_loss_pct,
        f"average_consumption_loss_pct_difference_{column}_{group_value}": (
            (average_consumption_loss_pct - baseline_avg_annual_consumption_loss_pct)
            / baseline_avg_annual_consumption_loss_pct
            if baseline_avg_annual_consumption_loss_pct != 0
            else 0.0
        ),
        f"average_reconstruction_rate_{column}_{group_value}": group_households[
            "reconstruction_rate"
        ].mean(),
        f"average_reconstruction_rate_difference_{column}_{group_value}": (
            (
                group_households["reconstruction_rate"].mean()
                - affected_households["reconstruction_rate"].mean()
            )
            / affected_households["reconstruction_rate"].mean()
            if affected_households["reconstruction_rate"].mean() != 0
            else 0.0
        ),
        f"r_consumption_based_{column}_{group_value}": resilience["consumption_based"],
        f"r_consumption_based_difference_{column}_{group_value}": (
            (resilience["consumption_based"] - baseline_resilience["consumption_based"])
            / baseline_resilience["consumption_based"]
            if baseline_resilience["consumption_based"] != 0
            else 0.0
        ),
        f"r_wellbeing_based_{column}_{group_value}": resilience["wellbeing_based"],
        f"r_wellbeing_based_difference_{column}_{group_value}": (
            (resilience["wellbeing_based"] - baseline_resilience["wellbeing_based"])
            / baseline_resilience["wellbeing_based"]
            if baseline_resilience["wellbeing_based"] != 0
            else 0.0
        ),
    }
