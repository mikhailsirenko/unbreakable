import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Union, Tuple


def split_households_by_affected_percentage(
    households: pd.DataFrame, affected_percentage: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the households DataFrame into affected and unaffected based on the given percentage.

    Args:
        households (pd.DataFrame): Original households DataFrame.
        affected_percentage (float): Percentage of population affected (0 to 1).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Unaffected households, Affected households
    """

    # Calculate the new weights
    households["unaffected_weight"] = households["household_weight"] * (
        1 - affected_percentage
    )
    households["affected_weight"] = households["household_weight"] * affected_percentage

    # Create the unaffected DataFrame
    unaffected = households.copy()
    unaffected["household_weight"] = unaffected["unaffected_weight"]

    # Create the affected DataFrame
    affected = households.copy()
    affected["household_weight"] = affected["affected_weight"]

    # Drop the temporary columns
    unaffected = unaffected.drop(["unaffected_weight", "affected_weight"], axis=1)
    affected = affected.drop(["unaffected_weight", "affected_weight"], axis=1)

    # Add a column to indicate affected status
    affected["is_affected"] = True
    unaffected["is_affected"] = False

    return unaffected, affected


def distribute_disaster_impact(
    households: pd.DataFrame,
    disaster_impact_params: Dict[str, Union[str, float, int]],
    dwelling_vulnerability_params: Dict[str, Union[float, int]],
    pml: float = None,
    pct_population_affected: float = None,
    random_seed: int = None,
) -> pd.DataFrame:
    """
    Distribute disaster impact among households based on either PML or affected percentage.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        disaster_impact_params (Dict[str, Union[str, float, int]]): Parameters for disaster impact.
        dwelling_vulnerability_params (Dict[str, Union[float, int]]): Parameters for dwelling vulnerability.
        pml (float, optional): Probable Maximum Loss. Defaults to None.
        pct_population_affected (float, optional): Percentage of affected population. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.
    Returns:
        pd.DataFrame: Updated households DataFrame with distributed impact.

    Raises:
        ValueError: If neither PML nor pct_population_affected is provided, or if an unsupported distribution is specified.
    """
    if pml is None and pct_population_affected is None:
        raise ValueError("Either PML or pct_population_affected must be provided.")

    np.random.seed(random_seed)

    if pct_population_affected is not None:
        # Define how vulnerability can change
        # Decrease by 0-80%
        min_change = 0.2
        max_change = 1.0

        households["vulnerability_change_factor"] = np.random.uniform(
            min_change, max_change, size=len(households)
        )

        # Save original v
        households["v_orig"] = households["v"].copy()

        # Disaster increases vulnerability
        households["v"] *= households["vulnerability_change_factor"]

        # Ensure that the vulnerability is within the specified thresholds
        max_threshold = dwelling_vulnerability_params.get("max_threshold", 0.9)
        min_threshold = dwelling_vulnerability_params.get("min_threshold", 0.2)
        households["v"] = np.clip(households["v"], min_threshold, max_threshold)

        # No need to calculate asset impact ratio
        households["asset_impact_ratio"] = None

    else:
        # Use the original approach with PML
        poverty_bias_factor = disaster_impact_params.get("poverty_bias_factor", 1.0)

        if poverty_bias_factor == "random":
            if disaster_impact_params.get("distribution", "uniform") == "uniform":
                min_bias = disaster_impact_params.get("min_bias", 0.5)
                max_bias = disaster_impact_params.get("max_bias", 1.5)
                poverty_bias_factor = np.random.uniform(min_bias, max_bias)
            else:
                raise ValueError("Only uniform distribution is supported.")

        households["poverty_bias_factor"] = np.where(
            households["is_poor"], poverty_bias_factor, 1.0
        )

        total_weighted_vulnerability = (
            households[["keff", "v", "poverty_bias_factor", "household_weight"]]
            .prod(axis=1)
            .sum()
        )

        households["asset_impact_ratio"] = (
            pml / total_weighted_vulnerability
        ) * households["poverty_bias_factor"]

    return households


def determine_affected_households(
    households: pd.DataFrame,
    pml: float,
    params: Dict[str, Union[float, int]],
    random_seed: int,
) -> pd.DataFrame:
    """
    Determine which households are affected by a disaster.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        pml (float): Probable Maximum Loss.
        params (Dict[str, Union[float, int]]): Parameters for determination.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: Updated households DataFrame with affected status and asset loss.

    Raises:
        ValueError: If total asset stock is less than PML or if no suitable mask is found.
    """
    np.random.seed(random_seed)

    acceptable_loss_margin = pml * params.get("acceptable_loss_margin ", 0.025)
    total_asset_value = (households["keff"] * households["household_weight"]).sum()

    if total_asset_value < pml:
        raise ValueError("Total asset stock is less than PML.")

    min_random_threshold = params.get("min_random_threshold", 0)
    max_random_threshold = params.get("max_random_threshold", 1)
    num_simulations = params.get("num_simulations", 10000)

    impact_simulations = (
        np.random.uniform(
            min_random_threshold,
            max_random_threshold,
            (num_simulations, households.shape[0]),
        )
        <= households["asset_impact_ratio"].values
    )

    simulated_losses = (
        impact_simulations
        * households[["keff", "v", "household_weight"]].values.prod(axis=1)
    ).sum(axis=1)

    valid_simulations = (simulated_losses >= pml - acceptable_loss_margin) & (
        simulated_losses <= pml + acceptable_loss_margin
    )

    if not np.any(valid_simulations):
        raise ValueError(
            f"Cannot find affected households in {num_simulations} simulations."
        )

    chosen_simulation = impact_simulations[np.argmax(valid_simulations)]

    households["is_affected"] = chosen_simulation
    households["asset_loss"] = np.where(
        households["is_affected"],
        households[["keff", "v", "household_weight"]].prod(axis=1),
        0,
    )

    total_asset_loss = households["asset_loss"].sum()
    if not (
        pml - acceptable_loss_margin <= total_asset_loss <= pml + acceptable_loss_margin
    ):
        raise ValueError(
            f"Total asset loss ({total_asset_loss}) is not within the acceptable range."
        )

    return households


def calculate_compound_impact(disaster_spec, disaster_impacts):
    """
    Calculate the compound impact of multiple disasters.

    Args:
        disaster_spec (list): List of dictionaries containing disaster specifications.
        disaster_impacts (pd.DataFrame): DataFrame containing disaster risk data.

    Returns:
        dict: Dictionary containing compound disaster impact.
    """
    # If disaster_spec len is 1 there are no coinciding disasters, return the disaster_risk as is
    if len(disaster_spec) == 1:
        # Disaster impacts have data on multiple return periods
        # Filter the disaster impacts for the return period in disaster_spec
        rp = disaster_spec[0].get("return_period")
        filtered_impacts = disaster_impacts if rp is None else disaster_impacts[disaster_impacts["rp"] == rp]
        return {disaster_spec[0]["event_time"]: filtered_impacts}

    # Group disasters by event_time
    event_time_dict = defaultdict(list)
    for disaster in disaster_spec:
        event_time_dict[disaster["event_time"]].append(disaster)

    # Filter coinciding events
    coinciding = {
        time: disasters
        for time, disasters in event_time_dict.items()
        if len(disasters) > 1
    }

    # Filter individual events
    individual = {
        time: disasters
        for time, disasters in event_time_dict.items()
        if len(disasters) == 1
    }

    # Iterate over coinciding disasters and calculate compound impact as a sum of PMLs
    compound_risk = {}
    for time, events in coinciding.items():
        disaster_types = set(event["type"] for event in events)
        return_periods = [event["return_period"] for event in events]
        compound_type = "+".join(sorted(disaster_types))
        compound_return_period = "+".join(sorted(return_periods))

        # Filter relevant risks once
        relevant_risks = disaster_impacts[
            (disaster_impacts["disaster_type"].isin(disaster_types))
            & (disaster_impacts["rp"].isin(return_periods))
        ]

        # Group by spatial_unit and calculate compound impact
        compound_impact = (
            relevant_risks.groupby("spatial_unit")
            .agg(
                {
                    "pml": "sum",
                    "loss_fraction": "sum",
                    "residential": "first",
                    "non_residential": "first",
                    "total_exposed_stock": "first",
                }
            )
            .reset_index()
        )

        # Ensure that loss_fraction is in [0, 1]
        compound_impact["loss_fraction"] = compound_impact["loss_fraction"].clip(0, 1)

        # Ensure that PML is not greater than total exposed stock
        compound_impact["pml"] = compound_impact[["pml", "total_exposed_stock"]].min(
            axis=1
        )

        # Name compound disaster
        compound_impact["disaster_type"] = compound_type
        compound_impact["rp"] = compound_return_period

        # Reorder columns
        sorted_columns = [
            "disaster_type",
            "spatial_unit",
            "residential",
            "non_residential",
            "total_exposed_stock",
            "rp",
            "pml",
            "loss_fraction",
        ]
        compound_risk[time] = compound_impact[sorted_columns]

    individual_risk = {}
    for time, events in individual.items():
        for event in events:
            disaster_type = event["type"]
            return_period = event["return_period"]

            current_risk = disaster_impacts[
                (disaster_impacts["disaster_type"] == disaster_type)
                & (disaster_impacts["rp"] == return_period)
            ].copy()

            individual_risk[time] = current_risk

    # Combine individual and compound risk dicts
    combined_risk = {**individual_risk, **compound_risk}

    # Sort combined risk by event time (its keys)
    combined_risk = dict(sorted(combined_risk.items()))

    return combined_risk


def calculate_years_until_next_event(disaster_impacts, params):
    """Calculate the time until the next event."""
    # Initialize a new dictionary to store the time until the next event
    time_until_next_event = {}
    sorted_times = sorted(disaster_impacts.keys())

    # If there is only one event, return max_years
    if len(disaster_impacts) == 1:
        return {sorted_times[0]: params["recovery_params"]["max_years"]}

    # Iterate through the sorted keys and calculate the difference
    for i in range(len(sorted_times) - 1):
        current_time = sorted_times[i]
        next_time = sorted_times[i + 1]
        time_until_next_event[current_time] = next_time - current_time

    # Handle the last event, which has no next event
    max_years = params["recovery_params"]["max_years"]
    time_until_next_event[sorted_times[-1]] = (
        max_years - time_until_next_event[sorted_times[-2]]
    )
    return time_until_next_event
