import pandas as pd
from typing import Dict


def analyze_poor_population(
    households: pd.DataFrame, max_years: int
) -> tuple[int, int, int, pd.DataFrame, pd.DataFrame]:
    """
    Analyze and categorize poor population based on initial state and after-disaster effects.

    This function identifies the initially poor population, the affected poor population,
    and the newly poor population after considering disaster effects over a specified recovery period.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        max_years (int): Number of years for calculating annualized consumption loss.

    Returns:
        tuple[int, int, int, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - initial_poor_count: Number of initially poor households.
            - new_poor_count: Number of newly poor households after disaster effects.
            - affected_poor_count: Number of initially poor households affected by the disaster.
            - initial_poor_df: DataFrame of initially poor households.
            - new_poor_df: DataFrame of newly poor households.

    Note:
        The function assumes the presence of specific columns in the input DataFrame:
        'poverty_line', 'is_poor', 'household_weight', 'ever_affected', 'exp', 'consumption_loss_npv'.
    """
    poverty_line = households["poverty_line"].iloc[0]

    # Identify initially poor households
    initial_poor_df = households[households["is_poor"]]
    initial_poor_count = round(initial_poor_df["household_weight"].sum())

    # Identify affected poor households
    affected_poor_df = initial_poor_df[initial_poor_df["ever_affected"]]
    affected_poor_count = round(affected_poor_df["household_weight"].sum())

    # Identify newly poor households
    initially_not_poor_df = households[~households["is_poor"]]
    affected_not_poor_df = initially_not_poor_df[initially_not_poor_df["ever_affected"]]

    post_disaster_expenditure = (
        affected_not_poor_df["exp"]
        - affected_not_poor_df["consumption_loss_npv"] / max_years
    )

    # ! The poverty line is yearly and the expenditure is monthly
    new_poor_df = affected_not_poor_df[
        post_disaster_expenditure < (poverty_line / 12)
    ].copy()
    new_poor_df["is_poor"] = True
    new_poor_count = round(new_poor_df["household_weight"].sum())

    return (
        initial_poor_count,
        new_poor_count,
        affected_poor_count,
        initial_poor_df,
        new_poor_df,
    )


def analyze_poverty_duration(
    households: pd.DataFrame, max_years: int
) -> Dict[int, int]:
    """
    Analyze the duration of poverty for affected households.

    This function calculates the number of people experiencing poverty for different
    durations, up to a specified maximum number of years.

    Args:
        households (pd.DataFrame): DataFrame containing affected household data.
            Must include 'weeks_in_poverty' and 'wgt' (weight) columns.
        max_years (int, optional): Maximum number of years to consider. Defaults to 10.

    Returns:
        Dict[int, int]: A dictionary where keys are years in poverty (0 to max_years),
            and values are the number of people experiencing poverty for that duration.

    Note:
        The function assumes that 'weeks_in_poverty' and 'household_weight' columns exist in the input DataFrame.
    """
    # Calculate years in poverty from weeks
    households = households.copy()
    households["years_in_poverty"] = households["weeks_in_poverty"] // 52

    # Initialize dictionary with zeros for all year counts
    poverty_duration_counts = {year: 0 for year in range(max_years + 1)}

    # Group by years in poverty and sum weights
    grouped_data = (
        households.groupby("years_in_poverty")["household_weight"]
        .sum()
        .round()
        .astype(int)
    )

    # Update dictionary with actual counts, capped at max_years
    poverty_duration_counts.update(grouped_data[grouped_data.index <= max_years])

    return poverty_duration_counts


# TODO: Review how the poverty gaps are calculated


def calculate_poverty_gaps(
    initial_poor: pd.DataFrame, new_poor: pd.DataFrame, max_years: int = 10
) -> tuple[float, float, float]:
    """
    Calculate the poverty gaps at the beginning and end of the simulation.
    This function computes three poverty gaps:
    1. Initial poverty gap
    2. Updated poverty gap for initially poor population
    3. Overall poverty gap including newly poor population

    Args:
        initial_poor (pd.DataFrame): DataFrame of initially poor population.
        new_poor (pd.DataFrame): DataFrame of newly poor population.
        max_years (int): Number of years for calculating annualized consumption loss. Defaults to 10.

    Returns:
        tuple[float, float, float]: A tuple containing:
            - initial_poverty_gap: Poverty gap at the beginning of the simulation.
            - updated_initial_poverty_gap: Updated poverty gap for initially poor population.
            - overall_poverty_gap: Poverty gap including both initial and newly poor populations.

    Raises:
        ValueError: If any calculated poverty gap is greater than 1.
    """

    def calculate_average_expenditure(df: pd.DataFrame) -> float:
        if df.empty:
            return 0
        return (df["exp"] * df["household_weight"]).sum() / df["household_weight"].sum()

    def calculate_poverty_gap(average_expenditure: float, poverty_line: float) -> float:
        if average_expenditure >= poverty_line:
            return 0
        return (poverty_line - average_expenditure) / poverty_line

    # Handle case when initial_poor is empty
    if initial_poor.empty:
        return 0, 0, 0 if new_poor.empty else 1

    poverty_line = initial_poor["poverty_line"].iloc[0]
    initial_poor = initial_poor.copy()
    new_poor = new_poor.copy()

    initial_avg_expenditure = calculate_average_expenditure(initial_poor)
    initial_poverty_gap = calculate_poverty_gap(initial_avg_expenditure, poverty_line)

    # Handle case when new_poor is empty
    if new_poor.empty:
        updated_initial_avg_expenditure = calculate_average_expenditure(
            initial_poor.assign(
                exp=lambda x: x["exp"] - x["consumption_loss_npv"] / max_years
            )
        )
        updated_initial_poverty_gap = calculate_poverty_gap(
            updated_initial_avg_expenditure, poverty_line
        )
        return (
            initial_poverty_gap,
            updated_initial_poverty_gap,
            updated_initial_poverty_gap,
        )

    all_poor = pd.concat([initial_poor, new_poor])
    all_poor["exp"] -= all_poor["consumption_loss_npv"] / max_years
    overall_avg_expenditure = calculate_average_expenditure(all_poor)
    overall_poverty_gap = calculate_poverty_gap(overall_avg_expenditure, poverty_line)

    initial_poor["exp"] -= initial_poor["consumption_loss_npv"] / max_years
    updated_initial_avg_expenditure = calculate_average_expenditure(initial_poor)
    updated_initial_poverty_gap = calculate_poverty_gap(
        updated_initial_avg_expenditure, poverty_line
    )

    if any(
        gap > 1
        for gap in [
            initial_poverty_gap,
            updated_initial_poverty_gap,
            overall_poverty_gap,
        ]
    ):
        raise ValueError("Poverty gap cannot be greater than 1")

    return initial_poverty_gap, updated_initial_poverty_gap, overall_poverty_gap
