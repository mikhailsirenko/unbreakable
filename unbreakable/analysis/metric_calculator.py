import pandas as pd
from typing import Dict, Any
import json
import numpy as np

from unbreakable.analysis.poverty_metrics import (
    analyze_poor_population,
    analyze_poverty_duration,
    calculate_poverty_gaps,
)
from unbreakable.analysis.resilience_metrics import calculate_socioeconomic_resilience
from unbreakable.analysis.economic_metrics import (
    calculate_average_annual_consumption_loss,
)
from unbreakable.analysis.distributional_metrics import (
    calculate_distributional_impacts,
)


def calculate_metrics(
    households: pd.DataFrame, max_years: int, analysis_params: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate various metrics across affected households.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        max_years (int): Number of years for calculating annualized consumption loss.
        analysis_params (Dict[str, Any]): Dictionary containing analysis parameters.

    Returns:
        Dict[str, float]: A dictionary containing calculated metrics.
    """
    affected_households = households[households["ever_affected"]].copy()

    # Calculate years to recover
    affected_households["years_to_recover"] = np.round(
        np.log(1 / 0.05) / affected_households["reconstruction_rate"]
    )

    avg_years_to_recover = affected_households["years_to_recover"].mean()
    avg_dwelling_vulnerability = affected_households["v"].mean()

    # Calculate poverty metrics
    (
        initial_poor_count,
        new_poor_count,
        affected_poor_count,
        initial_poor_df,
        new_poor_df,
    ) = analyze_poor_population(households, max_years)

    poverty_duration = analyze_poverty_duration(affected_households, max_years)

    initial_poverty_gap, updated_initial_poverty_gap, overall_poverty_gap = (
        calculate_poverty_gaps(initial_poor_df, new_poor_df, max_years)
    )

    # Calculate resilience metrics
    resilience = calculate_socioeconomic_resilience(affected_households)

    # Calculate economic metrics
    avg_annual_consumption_loss, avg_annual_consumption_loss_pct = (
        calculate_average_annual_consumption_loss(affected_households, max_years)
    )

    # Compile results
    results = {
        "households_count": len(households),
        "affected_households_count": len(affected_households),
        "people_count": households["household_weight"].sum(),
        "affected_people_count": affected_households["household_weight"].sum(),
        "initial_poor_count": initial_poor_count,
        "new_poor_count": new_poor_count,
        "affected_poor_count": affected_poor_count,
        "initial_poverty_gap": initial_poverty_gap,
        "updated_initial_poverty_gap": updated_initial_poverty_gap,
        "overall_poverty_gap": overall_poverty_gap,
        "resilience_consumption_based": resilience["consumption_based"],
        "resilience_wellbeing_based": resilience["wellbeing_based"],
        "annual_average_consumption_loss": avg_annual_consumption_loss,
        "annual_average_consumption_loss_pct": avg_annual_consumption_loss_pct,
        "average_years_to_recover": avg_years_to_recover,
        "average_dwelling_vulnerability": avg_dwelling_vulnerability,
        **{
            f"poverty_duration_{year}": count
            for year, count in poverty_duration.items()
        },
    }

    # Calculate distributional impacts if required
    if analysis_params.get("distributional_impacts", False):
        distributional_outcomes = calculate_distributional_impacts(
            households,
            max_years,
            analysis_params.get("socioeconomic_attributes", []),
            avg_annual_consumption_loss_pct,
            resilience,
        )
        results.update(distributional_outcomes)

    # Save results keys in a json file
    with open("experiments/outcome_names_ordered.json", "w") as f:
        json.dump(list(results.keys()), f)

    return results
