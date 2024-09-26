import pandas as pd
import numpy as np
import os
import pickle
from typing import Dict, List


def estimate_socioeconomic_impact(
    households: pd.DataFrame,
    params: dict,
    years_until_next_event: int,
    disaster_type: str,
    return_period: str = None,
    random_seed: int = None,
) -> pd.DataFrame:
    """
    Estimate the socioeconomic impact of a disaster on households.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        params (dict): Dictionary containing economic, disaster impact, and analysis parameters.
        years_until_next_event (int): Number of years until the next disaster event.
        disaster_type (str): Type of disaster.
        return_period (str): Return period of the disaster.
        random_seed (int): Random seed used in the simulation. Note that nothing is stochastic in this function.

    Returns:
        pd.DataFrame: Updated household data with estimated socioeconomic impact.
    """
    # TODO: Review how the socioeconomic impact is estimated in case of consecutive events

    # Extract parameters
    economic_params = params["economic_params"]
    analysis_params = params["analysis_params"]
    disaster_impact_params = params["disaster_params"]["disaster_impact_params"]

    # Set up constants
    weeks_in_year = 52
    dt = 1 / weeks_in_year

    # Use this columns to store cumulative losses across multiple events
    columns_to_update: List[str] = [
        "consumption_loss",
        "consumption_loss_npv",
        "net_consumption_loss",
        "net_consumption_loss_npv",
        "weeks_in_poverty",
        "wellbeing_loss",
    ]

    # These columns must be reset to zero for each event
    columns_to_reset: List[str] = ["c_t", "c_t_unaffected"]

    households = households.copy()
    households[columns_to_update] = households.get(columns_to_update, 0.0)
    households[columns_to_reset] = 0.0

    # Filter affected households
    affected_households = households[households["is_affected"] == True].copy()

    if affected_households.empty:
        return households

    # Dictionary to store consumption recovery data
    consumption_recovery = {}

    for t in np.linspace(
        0, years_until_next_event, int(years_until_next_event * weeks_in_year)
    ):
        update_affected_households(
            affected_households, t, economic_params, disaster_impact_params, dt
        )

        if analysis_params["save_consumption_recovery"]:
            store_consumption_recovery(consumption_recovery, affected_households, t)

    # Update main dataframe
    households.loc[affected_households.index, columns_to_update + columns_to_reset] = (
        affected_households[columns_to_update + columns_to_reset]
    )

    # Save consumption recovery data if required
    if analysis_params["save_consumption_recovery"]:
        save_consumption_recovery(
            consumption_recovery,
            params["country"],
            households["spatial_unit"].values[0],
            disaster_type,
            return_period,
            random_seed,
        )

    return households


def update_affected_households(
    affected_households: pd.DataFrame,
    t: float,
    economic_params: dict,
    disaster_impact_params: dict,
    dt: float,
):
    """
    Update the affected households' data for a given time step.

    Args:
        affected_households (pd.DataFrame): DataFrame of affected households.
        t (float): Current time step.
        economic_params (dict): Dictionary of economic parameters.
        disaster_impact_params (dict): Dictionary of disaster impact parameters.
        dt (float): Time step size.
    """
    exponential_multiplier = np.exp(-affected_households["reconstruction_rate"] * t)
    growth_factor = (1 + economic_params["income_and_expenditure_growth"]) ** t

    expenditure = growth_factor * affected_households["exp"]
    savings = growth_factor * affected_households["sav"]
    asset_loss = (
        growth_factor
        * affected_households["v"]
        * affected_households["keff"]
        * affected_households["reconstruction_rate"]
    )
    income_loss = (
        growth_factor
        * economic_params["average_productivity"]
        * affected_households["keff"]
        * affected_households["v"]
    )

    if disaster_impact_params["add_income_loss"]:
        affected_households["c_t"] = expenditure + exponential_multiplier * (
            savings - asset_loss - income_loss
        )
    else:
        affected_households["c_t"] = expenditure + exponential_multiplier * (
            savings - asset_loss
        )

    affected_households["c_t_unaffected"] = expenditure

    # Ensure c_t is at least 1
    # TODO Review this clipping, maybe it should be 0
    affected_households["c_t"] = (
        affected_households[["c_t", "c_t_unaffected"]].min(axis=1).clip(lower=0)
    )

    update_losses(affected_households, t, economic_params, dt)


def update_losses(
    affected_households: pd.DataFrame, t: float, economic_params: dict, dt: float
):
    """
    Update the losses for affected households.

    Args:
        affected_households (pd.DataFrame): DataFrame of affected households.
        t (float): Current time step.
        economic_params (dict): Dictionary of economic parameters.
        dt (float): Time step size.
    """
    # ! If c_t was larger than c_t_unaffected,
    # the consumption_loss will be 0
    # TODO Fix this

    consumption_loss = dt * (
        affected_households["c_t_unaffected"] - affected_households["c_t"]
    )

    consumption_loss_npv = consumption_loss * np.exp(
        -economic_params["discount_rate"] * t
    )

    affected_households["consumption_loss"] += consumption_loss
    affected_households["consumption_loss_npv"] += consumption_loss_npv

    net_consumption_loss = (
        dt
        * np.exp(-affected_households["reconstruction_rate"] * t)
        * affected_households["v"]
        * (1 + economic_params["income_and_expenditure_growth"]) ** t
        * affected_households["rent"]
    )
    net_consumption_loss_npv = net_consumption_loss * np.exp(
        -economic_params["discount_rate"] * t
    )

    affected_households["net_consumption_loss"] += net_consumption_loss
    affected_households["net_consumption_loss_npv"] += net_consumption_loss_npv

    # ! Poverty line is yearly and c_t is weekly
    affected_households.loc[
        affected_households["c_t"] < (affected_households["poverty_line"] / 52),
        "weeks_in_poverty",
    ] += 1

    update_wellbeing(affected_households, t, economic_params, dt)


def update_wellbeing(
    affected_households: pd.DataFrame, t: float, economic_params: dict, dt: float
):
    """
    Update the wellbeing loss for affected households.

    Args:
        affected_households (pd.DataFrame): DataFrame of affected households.
        t (float): Current time step.
        economic_params (dict): Dictionary of economic parameters.
        dt (float): Time step size.
    """
    unaffected_utility = (
        affected_households["c_t_unaffected"]
        ** (1 - economic_params["consumption_utility"])
    ) / (1 - economic_params["consumption_utility"])

    adjusted_wellbeing = (
        (
            1
            - (
                (affected_households["c_t_unaffected"] - affected_households["c_t"])
                / affected_households["c_t_unaffected"]
            )
            * np.exp(-affected_households["reconstruction_rate"] * t)
        )
        ** (1 - economic_params["consumption_utility"])
    ) - 1

    wellbeing_loss = (
        unaffected_utility
        * dt
        * adjusted_wellbeing
        * np.exp(-economic_params["discount_rate"] * t)
    )

    # If t = 0, and wellbeing_loss is inf or nan, turn it into 0
    if t == 0:
        wellbeing_loss = wellbeing_loss.replace([np.inf, -np.inf], 0).fillna(0)

    affected_households["wellbeing_loss"] += wellbeing_loss


def store_consumption_recovery(
    consumption_recovery: Dict, affected_households: pd.DataFrame, t: float
):
    """
    Store consumption recovery data for a given time step.

    Args:
        consumption_recovery (Dict): Dictionary to store consumption recovery data.
        affected_households (pd.DataFrame): DataFrame of affected households.
        t (float): Current time step.
    """
    consumption_recovery[t] = affected_households[
        [
            "household_id",
            "household_weight",
            "is_poor",
            "reconstruction_rate",
            "c_t_unaffected",
            "c_t",
        ]
    ].set_index("household_id")


def save_consumption_recovery(
    consumption_recovery: Dict,
    country: str,
    spatial_unit: str,
    disaster_type: str,
    return_period: int = None,
    random_seed: int = None,
):
    """
    Save consumption recovery to a file.

    Args:
        consumption_recovery (Dict): Dictionary containing consumption recovery data.
        country (str): Country name.
        spatial_unit (str): Spatial unit identifier.
        disaster_type (str): Type of disaster.
        return_period (int): Return period of the disaster.
        random_seed (int): Random seed used in the simulation.
    """
    folder = f"../results/{country}/consumption_recovery/"
    os.makedirs(folder, exist_ok=True)

    with open(
        f"{folder}/{spatial_unit}_{disaster_type}_rp={return_period}_{random_seed}.pickle",
        "wb",
    ) as handle:
        pickle.dump(consumption_recovery, handle, protocol=pickle.HIGHEST_PROTOCOL)


# def update_wellbeing(
#     affected_households: pd.DataFrame, t: float, economic_params: dict, dt: float
# ):
#     """
#     Update the wellbeing loss for affected households with added diagnostic checks.

#     Args:
#         affected_households (pd.DataFrame): DataFrame of affected households.
#         t (float): Current time step.
#         economic_params (dict): Dictionary of economic parameters.
#         dt (float): Time step size.
#     """
#     # Check 1: Print input parameters
#     print(f"Time step: {t}, dt: {dt}")
#     print(f"Economic params: {economic_params}")

#     # Check 2: Verify c_t_unaffected is positive
#     if (affected_households["c_t_unaffected"] <= 0).any():
#         print("Warning: c_t_unaffected contains non-positive values")
#         print(affected_households[affected_households["c_t_unaffected"] <= 0])

#     unaffected_utility = (
#         affected_households["c_t_unaffected"]
#         ** (1 - economic_params["consumption_utility"])
#     ) / (1 - economic_params["consumption_utility"])

#     # Check 3: Verify unaffected_utility calculation
#     if np.isinf(unaffected_utility).any() or np.isnan(unaffected_utility).any():
#         print("Warning: unaffected_utility contains inf or nan values")
#         print(
#             affected_households[
#                 np.isinf(unaffected_utility) | np.isnan(unaffected_utility)
#             ]
#         )

#     relative_consumption_change = (
#         affected_households["c_t_unaffected"] - affected_households["c_t"]
#     ) / affected_households["c_t_unaffected"]

#     # Check 4: Verify relative consumption change
#     if (relative_consumption_change >= 1).any() or (
#         relative_consumption_change < 0
#     ).any():
#         print("Warning: relative consumption change is >= 1 or < 0")
#         print(
#             affected_households[
#                 (relative_consumption_change >= 1) | (relative_consumption_change < 0)
#             ]
#         )

#     reconstruction_term = np.exp(-affected_households["reconstruction_rate"] * t)

#     # Check 5: Verify reconstruction term
#     if np.isinf(reconstruction_term).any() or np.isnan(reconstruction_term).any():
#         print("Warning: reconstruction term contains inf or nan values")
#         print(
#             affected_households[
#                 np.isinf(reconstruction_term) | np.isnan(reconstruction_term)
#             ]
#         )

#     adjusted_wellbeing = (
#         (1 - relative_consumption_change * reconstruction_term)
#         ** (1 - economic_params["consumption_utility"])
#     ) - 1

#     # Check 6: Verify adjusted_wellbeing calculation
#     if np.isinf(adjusted_wellbeing).any() or np.isnan(adjusted_wellbeing).any():
#         print("Warning: adjusted_wellbeing contains inf or nan values")
#         print(
#             affected_households[
#                 np.isinf(adjusted_wellbeing) | np.isnan(adjusted_wellbeing)
#             ]
#         )

#     discount_term = np.exp(-economic_params["discount_rate"] * t)

#     wellbeing_loss = unaffected_utility * dt * adjusted_wellbeing * discount_term

#     # Check 7: Verify final wellbeing_loss calculation
#     if np.isinf(wellbeing_loss).any() or np.isnan(wellbeing_loss).any():
#         print("Warning: wellbeing_loss contains inf or nan values")
#         print(affected_households[np.isinf(wellbeing_loss) | np.isnan(wellbeing_loss)])

#     # If t = 0, and wellbeing_loss is inf or nan, turn it into 0
#     if t == 0:
#         wellbeing_loss = wellbeing_loss.replace([np.inf, -np.inf], 0).fillna(0)

#     affected_households["wellbeing_loss"] += wellbeing_loss

#     # Check 8: Verify final wellbeing_loss in DataFrame
#     if (
#         np.isinf(affected_households["wellbeing_loss"]).any()
#         or np.isnan(affected_households["wellbeing_loss"]).any()
#     ):
#         print("Warning: Final wellbeing_loss contains inf or nan values")
#         print(
#             affected_households[
#                 np.isinf(affected_households["wellbeing_loss"])
#                 | np.isnan(affected_households["wellbeing_loss"])
#             ]
#         )

#     return affected_households
