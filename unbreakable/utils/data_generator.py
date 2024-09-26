import numpy as np
import pandas as pd
from typing import Dict, Literal

# TODO: Generate example population impact data

ROOF_SCORES: Dict[str, float] = {
    "Concrete": 0.2,
    "Finished – Concrete": 0.2,
    "Finished – Metal": 0.4,
    "Tile": 0.5,
    "Shingle (asphalt)": 0.5,
    "Shingle (other)": 0.5,
    "Finished – Asbestos": 0.6,
    "Shingle (wood)": 0.6,
    "Sheet metal (galvanize, galvalume)": 0.65,
    "Makeshift/thatched": 0.8,
    "Natural – Thatch/palm leaf": 0.8,
    "Other": 0.8,
}

WALL_SCORES: Dict[str, float] = {
    "Concrete/Concrete blocks": 0.2,
    "Concrete/Concrete Blocks": 0.2,
    "Finished – Cement blocks": 0.3,
    "Brick/Blocks": 0.35,
    "Wood & Concrete": 0.4,
    "Finished – Stone with lime/cement": 0.45,
    "Finished – GRC/Gypsum/Asbestos": 0.5,
    "Wood/Timber": 0.6,
    "Natural – Other": 0.6,
    "Plywood": 0.7,
    "Rudimentary – Plywood": 0.7,
    "Makeshift": 0.8,
    "Other": 0.8,
}

FLOOR_SCORES: Dict[str, float] = {
    "Finished – Cement/red bricks": 0.2,
    "Finished – Parquet or polished wood": 0.4,
    "Rudimentary – Wood planks": 0.5,
    "Natural – Earth/sand": 0.7,
    "Other": 0.6,
}


def generate_households(
    num_households: int, num_spatial_units: int, seed: int = 0
) -> pd.DataFrame:
    """
    Generate dummy households with various economic and demographic attributes based on income.

    Args:
        num_households (int): Number of households to generate.
        num_spatial_units (int): Number of spatial units where households are located.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing household data.
    """
    np.random.seed(seed)

    # Constants
    income_to_expenditure_ratio = 1.2
    housing_expenditure_ratio = 0.3
    effective_capital_stock_factor = 1 / 0.35
    home_ownership_rate = 0.7
    min_savings_rate = 0.05

    # Generate base income data
    income = np.random.normal(2000, 500, num_households)
    income = np.clip(income, 1000, 3000)

    # Generate additional columns based on income
    female_headed = np.random.rand(num_households) < (
        0.3 - (income - 1000) / 2000 * 0.2
    )
    urban = np.random.rand(num_households) < ((income - 1000) / 2000 * 0.6 + 0.2)

    # Determine literacy levels based on income
    literacy_thresholds = np.percentile(income, [33, 66])
    literacy = np.select(
        [
            income <= literacy_thresholds[0],
            (income > literacy_thresholds[0]) & (income <= literacy_thresholds[1]),
            income > literacy_thresholds[1],
        ],
        ["Low", "Medium", "High"],
        default="Medium",
    )

    owns_home = np.random.rand(num_households) < (
        (income - 1000) / 2000 * 0.5 + home_ownership_rate - 0.25
    )

    # Assign spatial units
    spatial_units = [f"region_{i}" for i in range(num_spatial_units)]
    assigned_spatial_units = np.random.choice(spatial_units, num_households)

    data = {
        "inc": income,
        "owns_home": owns_home,
        "keff": income * effective_capital_stock_factor,
        "spatial_unit": assigned_spatial_units,
        "female_headed": female_headed,
        "urban": urban,
        "literacy": literacy,
    }

    data["roof"] = [select_material(inc, ROOF_SCORES) for inc in income]
    data["walls"] = [select_material(inc, WALL_SCORES) for inc in income]
    data["floor"] = [select_material(inc, FLOOR_SCORES) for inc in income]

    # Calculate housing expenditure
    data["exp_house"] = np.where(
        data["owns_home"], 0, income * housing_expenditure_ratio
    )

    # Calculate total expenditure ensuring minimum savings
    max_expenditure = income * (1 - min_savings_rate)
    data["exp"] = np.minimum(
        income / income_to_expenditure_ratio, max_expenditure - data["exp_house"]
    )

    # Calculate savings
    data["sav"] = income - data["exp"] - data["exp_house"]

    # Assign each household a unique ID and weight
    data["household_id"] = range(1, num_households + 1)
    data["household_weight"] = np.random.uniform(1, 100, num_households)

    # Define poverty line and poor status
    poverty_line = np.percentile(data["inc"], 20)
    data["poverty_line"] = poverty_line
    data["is_poor"] = data["inc"] < poverty_line

    # Sort columns
    sorted_columns = [
        "spatial_unit",
        "household_id",
        "household_weight",
        "inc",
        "exp",
        "sav",
        "owns_home",
        "exp_house",
        "keff",
        "roof",
        "walls",
        "floor",
        "poverty_line",
        "is_poor",
        "female_headed",
        "urban",
        "literacy",
    ]

    # Assert conditions
    assert (
        data["inc"] >= data["exp"] + data["exp_house"]
    ).all(), "Income should be greater than expenditure"
    assert (data["sav"] >= 0).all(), "Savings should be positive"
    assert (data["exp"] >= 0).all(), "Expenditure should be positive"
    assert (data["exp_house"] >= 0).all(), "Housing expenditure should be positive"

    return pd.DataFrame(data)[sorted_columns]


def select_material(income: float, material_scores: Dict[str, float]) -> str:
    """
    Select a material based on income, favoring lower vulnerability scores for higher incomes.

    Args:
        income (float): Household income
        material_scores (Dict[str, float]): Dictionary of materials and their vulnerability scores

    Returns:
        str: Selected material
    """
    materials = list(material_scores.keys())
    scores = np.array(list(material_scores.values()))

    # Invert and normalize scores to use as probabilities
    inv_scores = 1 - scores / scores.max()
    probabilities = inv_scores / inv_scores.sum()

    # Adjust probabilities based on income
    income_factor = min(income / 3000, 1)  # Normalize income, cap at 3000
    adjusted_probs = probabilities * (1 + income_factor)
    adjusted_probs /= adjusted_probs.sum()

    return np.random.choice(materials, p=adjusted_probs)


def generate_disaster_risk(disaster_type: str, num_spatial_units: int):
    """Generate dummy disaster data for a given disaster type and number of spatial units."""
    np.random.seed(42)  # For reproducibility

    spatial_units = [f"region_{i}" for i in range(num_spatial_units)]
    return_periods = [10, 50, 100, 250]  # Common return periods

    data = []

    for rp in return_periods:
        rp_factor = rp / 250  # Normalize return period

        for spatial_unit in spatial_units:
            residential = np.random.uniform(100_000_000, 250_000_000)
            non_residential = np.random.uniform(50_000_000, 100_000_000)
            total = residential + non_residential

            # Adjust PML and loss fraction based on disaster type and return period
            if disaster_type == "flood":
                base_loss = 0.05
                max_loss = 0.3
            elif disaster_type == "hurricane":
                base_loss = 0.1
                max_loss = 0.4
            elif disaster_type == "earthquake":
                base_loss = 0.15
                max_loss = 0.5
            else:
                raise ValueError(f"Invalid disaster type: {disaster_type}")

            # Calculate loss fraction ensuring it increases with return period
            loss_fraction = base_loss + (max_loss - base_loss) * rp_factor
            # Add some randomness
            loss_fraction += np.random.uniform(-0.05, 0.05)
            loss_fraction = max(
                base_loss, min(max_loss, loss_fraction)
            )  # Ensure within bounds

            pml = total * loss_fraction

            data.append(
                {
                    "disaster_type": disaster_type,
                    "spatial_unit": spatial_unit,
                    "residential": round(residential, 2),
                    "non_residential": round(non_residential, 2),
                    "total_exposed_stock": round(total, 2),
                    "rp": rp,
                    "pml": round(pml, 2),
                    "loss_fraction": round(loss_fraction, 3),
                }
            )

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Generate dummy disaster risk and household survey data
    disaster_types = ["flood", "hurricane", "earthquake"]
    for disaster in disaster_types:
        df = generate_disaster_risk(disaster, num_spatial_units=64)
        df.to_csv(
            f"../../data/processed/asset_impacts/Example/{disaster}.csv", index=False
        )
    households = generate_households(100, num_spatial_units=64)
    households.to_csv("../../data/processed/household_survey/Example.csv", index=False)
