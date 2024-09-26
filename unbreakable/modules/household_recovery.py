import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_household_reconstruction_rates(
    households: pd.DataFrame,
    economic_params: Dict[str, float],
    recovery_params: Dict[str, float],
    use_precomputed_reconstruction_rates: bool = False,
    reconstruction_rates: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Calculate reconstruction rates for affected households.

    Args:
        households (pd.DataFrame): DataFrame containing household data.
        economic_params (Dict[str, float]): Economic parameters for reconstruction calculation.
        recovery_params (Dict[str, float]): Recovery parameters for reconstruction calculation.
        use_precomputed_reconstruction_rates (bool): Whether to use precomputed reconstruction rates.
        reconstruction_rates (Optional[pd.DataFrame]): Precomputed reconstruction rates data for 'population' impact data type.

    Returns:
        pd.DataFrame: Updated DataFrame with reconstruction rates for affected households.

    Raises:
        ValueError: If reconstruction rates are invalid or if precomputed rates are missing for 'population' type.
    """
    if use_precomputed_reconstruction_rates:
        if reconstruction_rates is None:
            raise ValueError(
                "Precomputed rates data must be provided for 'population' impact data type."
            )

        households["reconstruction_rate"] = 0.0

        # Identify affected households
        affected_mask = households["is_affected"]

        # NOTE Precomputed rates are calculated for rounded vulnerability values
        households.loc[affected_mask, "v_rounded"] = households.loc[
            affected_mask, "v"
        ].round(2)

        # Merge precomputed rates only for affected households
        affected_households = households.loc[affected_mask].merge(
            reconstruction_rates,
            left_on="v_rounded",
            right_on="v",
            how="left",
            suffixes=("", "_precomputed"),
        )

        # Check for missing precomputed rates
        missing_rates = affected_households["reconstruction_rate_precomputed"].isnull()
        if missing_rates.any():
            missing_v = affected_households.loc[missing_rates, "v_rounded"].unique()
            raise ValueError(
                f"Precomputed reconstruction rates not found for vulnerability values: {missing_v}"
            )

        # Update reconstruction rates for affected households
        households.loc[affected_mask, "reconstruction_rate"] = affected_households[
            "reconstruction_rate_precomputed"
        ].values

        # Clean up temporary columns
        households = households.drop(
            ["v_rounded", "v_precomputed", "reconstruction_rate_precomputed"],
            axis=1,
            errors="ignore",
        )

        return households

    else:
        # Original calculation for PML-based approach
        affected_mask = households["is_affected"]
        households.loc[affected_mask, "reconstruction_rate"] = 0.0
        households.loc[affected_mask, "reconstruction_rate"] = households.loc[
            affected_mask
        ].apply(
            lambda row: find_optimal_reconstruction_rate(
                row["v"], economic_params, recovery_params
            ),
            axis=1,
        )

    # Validate reconstruction rates
    validate_reconstruction_rates(
        households.loc[households["is_affected"], "reconstruction_rate"],
        recovery_params["max_years"],
    )

    return households


def find_optimal_reconstruction_rate(
    dwelling_vulnerability: float,
    economic_params: Dict[str, float],
    recovery_params: Dict[str, float],
) -> float:
    """
    Find the optimal reconstruction rate for a household given its dwelling vulnerability.

    Args:
        dwelling_vulnerability (float): Dwelling vulnerability score of the household.
        economic_params (Dict[str, float]): Economic parameters for reconstruction calculation.
        recovery_params (Dict[str, float]): Recovery parameters for reconstruction calculation.

    Returns:
        float: Optimal reconstruction rate.

    Notes:
        This function uses a numerical optimization approach to find the optimal
        reconstruction rate that maximizes utility over time.
    """
    max_years = recovery_params["max_years"]
    lambda_increment = recovery_params["lambda_increment"]
    average_productivity = economic_params["average_productivity"]
    consumption_utility = economic_params["consumption_utility"]
    discount_rate = economic_params["discount_rate"]
    total_weeks = 52 * recovery_params["max_years"]
    dt = 1 / 52
    lambda_value = 0
    last_derivative_lambda = 0

    while True:
        derivative_lambda = 0
        for time in np.linspace(0, max_years, total_weeks):
            factor = average_productivity + lambda_value
            marginal_utility_term = (
                average_productivity
                - factor * dwelling_vulnerability * np.exp(-lambda_value * time)
            ) ** (-consumption_utility)
            time_productivity_adjustment = time * factor - 1
            discount_factor = np.exp(-time * (discount_rate + lambda_value))
            derivative_lambda += (
                marginal_utility_term
                * time_productivity_adjustment
                * discount_factor
                * dt
            )
        if (
            (last_derivative_lambda < 0 and derivative_lambda > 0)
            or (last_derivative_lambda > 0 and derivative_lambda < 0)
            or lambda_value > max_years
        ):
            return lambda_value
        last_derivative_lambda = derivative_lambda
        lambda_value += lambda_increment


def validate_reconstruction_rates(rates: pd.Series, max_years: int) -> None:
    """
    Validate the calculated reconstruction rates.

    Args:
        rates (pd.Series): Series of reconstruction rates.
        max_years (int): Maximum allowed years for reconstruction.

    Raises:
        ValueError: If any reconstruction rate is invalid (0 or equal to max_years).
    """
    if rates.eq(max_years).any():
        raise ValueError("Reconstruction rate not found for some households")
    if rates.eq(0).any():
        raise ValueError("Reconstruction rate is 0 for some households")
    if rates.isna().any():
        raise ValueError("Reconstruction rate is missing for some households")


def precompute_reconstruction_rates(
    economic_params, recovery_params, dwelling_vulnerabilities
):
    """Precompute optimal reconstruction rates for a range of dwelling vulnerabilities.
    Args:
        economic_params (Dict[str, float]): Economic parameters for reconstruction calculation.
        recovery_params (Dict[str, float]): Recovery parameters for reconstruction calculation.
        dwelling_vulnerabilities (np.ndarray): Array of dwelling vulnerabilities.
    Returns:
        None
    """
    optimal_rates = [
        find_optimal_reconstruction_rate(v, economic_params, recovery_params)
        for v in dwelling_vulnerabilities
    ]
    optimal_rates_df = pd.DataFrame(
        {
            "v": dwelling_vulnerabilities,
            "reconstruction_rate": optimal_rates,
        }
    )

    # Round vulnerabilities to 2 decimal places to ensure 0.01 step
    optimal_rates_df["v"] = optimal_rates_df["v"].round(2)

    return optimal_rates_df


if __name__ == "__main__":
    # NOTE These are dummy parameters for the sake of the example
    economic_params = {
        "average_productivity": 0.25636,
        "consumption_utility": 1.5,
        "discount_rate": 0.04,
    }
    recovery_params = {"max_years": 10, "lambda_increment": 0.01}

    # Generate vulnerabilities with exact 0.01 step
    dwelling_vulnerabilities = np.round(np.arange(0.2, 0.91, 0.01), 2)

    optimal_rates = precompute_reconstruction_rates(
        economic_params, recovery_params, dwelling_vulnerabilities
    )

    # Save precomputed rates to CSV
    country_name = "Example"

    optimal_rates.to_csv(
        f"../../data/generated/{country_name}/optimal_reconstruction_rates.csv",
        index=False,
    )
