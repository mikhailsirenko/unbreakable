import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from unbreakable.utils.data_loader import load_data, load_reconstruction_rates
from unbreakable.utils.household_data_preprocessor import prepare_household_data
from unbreakable.utils.data_randomizer import randomize_dwelling_vulnerability
from unbreakable.modules.dwelling_vulnerability import calculate_dwelling_vulnerability
from unbreakable.modules.disaster_impact import (
    calculate_compound_impact,
    calculate_years_until_next_event,
    distribute_disaster_impact,
    determine_affected_households,
    split_households_by_affected_percentage,
)
from unbreakable.modules.household_recovery import (
    calculate_household_reconstruction_rates,
)
from unbreakable.modules.socioeconomic_impact import estimate_socioeconomic_impact
from unbreakable.analysis.metric_calculator import calculate_metrics


def model(**params) -> dict:
    random_seed = params["random_seed"]

    # ------------------------- 1: Load and prepare data ------------------------- #
    params["current_policy"] = params.get("current_policy", "none")
    households, disaster_impacts = load_data(params)
    reconstruction_rates = load_reconstruction_rates(params)
    households = prepare_household_data(
        households, disaster_impacts, params, random_seed
    )

    # ------- 2: If compound events/multiple events update disaster impacts ------ #
    disaster_impacts = calculate_compound_impact(
        params["disaster_spec"], disaster_impacts
    )
    years_until_next_event = calculate_years_until_next_event(disaster_impacts, params)

    # ----------------- 3: Simulate disaster in each spatial unit ---------------- #
    outcomes = {}
    for spatial_unit in params["spatial_units"]:
        previous_disaster_type = None
        spatial_unit_households = households[
            households["spatial_unit"] == spatial_unit
        ].copy()

        # In case of multiple events keep track of cumulative impact
        spatial_unit_households["cumulative_asset_impact_ratio"] = 0
        spatial_unit_households["ever_affected"] = False

        # -------------------- 4: Iterate over each disaster event ------------------- #
        for event_time, current_impact in disaster_impacts.items():
            spatial_unit_households = simulate_disaster(
                spatial_unit_households,
                current_impact[current_impact["spatial_unit"] == spatial_unit],
                previous_disaster_type,
                params,
                reconstruction_rates,
                random_seed,
            )

            # Store the disaster type for the next event
            previous_disaster_type = current_impact[
                current_impact["spatial_unit"] == spatial_unit
            ]["disaster_type"].values[0]

            # ----------------------- 5. Apply policy interventions ---------------------- #
            # Note that adaptive social protection policies can be applied here
            # But retrofitting policies should be applied before the disaster event

            # --------------- 6. Estimate consumption and well-being losses -------------- #
            spatial_unit_households = estimate_socioeconomic_impact(
                spatial_unit_households,
                params,
                years_until_next_event[event_time],
                current_impact["disaster_type"].values[0],
                (
                    current_impact["rp"].iloc[0]
                    if "rp" in current_impact.columns and not current_impact.empty
                    else None
                ),
                random_seed,
            )

            # Update cumulative impact
            spatial_unit_households[
                "cumulative_asset_impact_ratio"
            ] += spatial_unit_households["asset_impact_ratio"]
            spatial_unit_households["ever_affected"] |= spatial_unit_households[
                "is_affected"
            ]
            # Reset affected status for next event
            spatial_unit_households["is_affected"] = False

        # ---------------- 7. Calculate outcomes for each spatial unit --------------- #
        outcomes[spatial_unit] = np.array(
            list(
                calculate_metrics(
                    spatial_unit_households,
                    params["recovery_params"]["max_years"],
                    params["analysis_params"],
                ).values()
            )
        )

    return outcomes


def simulate_disaster(
    households: pd.DataFrame,
    disaster_impact: Dict[str, Any],
    previous_disaster_type: str,
    params: Dict[str, Any],
    reconstruction_rates: Optional[pd.DataFrame] = None,
    random_seed: int = None,
) -> pd.DataFrame:
    """Simulate disaster impact and recovery for a single event.

    This function simulates the impact of a disaster on households and estimates subsequent
    recovery. It handles different types of disasters and can use either percentage of
    population affected or Probable Maximum Loss (PML) to determine the disaster's impact.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        disaster_impact (Dict[str, Any]): Dictionary containing disaster impact information.
            Must include 'disaster_type' and either 'pct_population_affected' or 'pml'.
        previous_disaster_type (str): The type of the previous disaster, used to determine
            if dwelling vulnerability should be randomized.
        params (Dict[str, Any]): Dictionary containing various simulation parameters.
            Must include 'dwelling_vulnerability_params', 'disaster_params', 'economic_params',
            and 'recovery_params'.
        precomputed_rates (Optional[pd.DataFrame], optional): Precomputed reconstruction rates.
            If None, rates will be calculated. Defaults to None.
        random_seed (int, optional): Seed for random number generation to ensure
            reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: Updated household data after simulating disaster impact and initial recovery.

    Raises:
        KeyError: If required keys are missing in the input dictionaries.
        ValueError: If neither 'pct_population_affected' nor 'pml' is provided in disaster_impact.

    Note:
        - The function modifies the input households DataFrame in-place.
        - The function assumes the existence of several helper functions like
          calculate_dwelling_vulnerability, randomize_dwelling_vulnerability, etc.
        - The disaster impact is distributed either based on percentage of population
          affected or PML, depending on which is provided in the disaster_impact dict.
    """
    # Extract disaster impact information
    disaster_type = disaster_impact["disaster_type"].values[0]

    # Two options are possible for the impact data type:
    # 'assets', then we have pml or 'population' then we have pct_population_affected
    pml = disaster_impact.get("pml", pd.Series([None])).iloc[0]
    pct_population_affected = disaster_impact.get(
        "pct_population_affected", pd.Series([None])
    ).iloc[0]

    # If the disaster type has changed, randomize the dwelling vulnerability
    if previous_disaster_type is None or previous_disaster_type != disaster_type:
        params["dwelling_vulnerability_params"]["randomize"] = True
    else:
        params["dwelling_vulnerability_params"]["randomize"] = False

    # 1: Calculate dwelling vulnerability given the disaster type and randomize if needed
    households = households.pipe(calculate_dwelling_vulnerability, disaster_type).pipe(
        randomize_dwelling_vulnerability,
        params["dwelling_vulnerability_params"],
        random_seed,
    )

    # 2.1: If impact data type is population,
    # distribute disaster impact based on percentage of population affected
    if pct_population_affected is not None:
        unaffected, affected = split_households_by_affected_percentage(
            households, pct_population_affected
        )

        affected = distribute_disaster_impact(
            affected,
            params["disaster_params"]["disaster_impact_params"],
            params["dwelling_vulnerability_params"],
            pml,
            pct_population_affected,
            random_seed,
        )

        households = pd.concat([unaffected, affected])

    # 2.2: If impact data type is assets, distribute disaster impact based on PML
    else:
        households = households.pipe(
            distribute_disaster_impact,
            params["disaster_params"]["disaster_impact_params"],
            params["dwelling_vulnerability_params"],
            pml,
            pct_population_affected,
            random_seed,
        ).pipe(
            determine_affected_households,
            pml,
            params["disaster_params"]["determine_affected_params"],
            random_seed,
        )

    # 3: Calculate/reuse precomputed household reconstruction rates
    households = calculate_household_reconstruction_rates(
        households,
        params["economic_params"],
        params["recovery_params"],
        params["recovery_params"]["use_precomputed_reconstruction_rates"],
        reconstruction_rates,
    )

    return households
