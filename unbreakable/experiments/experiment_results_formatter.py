import pandas as pd
import numpy as np
import json
import ast
from typing import Tuple, List, Dict, Any


def format_experiment_results(
    results: Tuple[pd.DataFrame, Dict[str, np.ndarray]],
    include_policies: bool = True,
    include_uncertainties: bool = True,
) -> pd.DataFrame:
    """
    Process experiment results from EMA Workbench format into a structured DataFrame.

    Args:
        results (Tuple[pd.DataFrame, Dict[str, np.ndarray]]): EMA Workbench experiment results.
        include_policies (bool): Whether to include policy values in the output.
        include_uncertainties (bool): Whether to include uncertainty values in the output.

    Returns:
        pd.DataFrame: Processed results as a structured DataFrame.
    """
    experiments, outcome_data = results
    outcome_names = load_outcome_names()

    base_columns = ["scenario", "policy", "spatial_unit", "random_seed"]
    policy_names = ["current_policy"]
    uncertainty_names = []  # Add uncertainty names if needed

    columns = build_column_list(
        base_columns,
        policy_names,
        uncertainty_names,
        outcome_names,
        include_policies,
        include_uncertainties,
    )

    processed_data = []
    for spatial_unit, spatial_unit_outcomes in outcome_data.items():
        spatial_unit_data = process_spatial_unit_data(
            experiments,
            spatial_unit_outcomes,
            spatial_unit,
            policy_names,
            uncertainty_names,
            outcome_names,
            include_policies,
            include_uncertainties,
        )
        processed_data.extend(spatial_unit_data)

    results_df = pd.DataFrame(processed_data, columns=columns)
    return results_df


def load_outcome_names() -> List[str]:
    """Load outcome names from JSON file."""
    with open("../../unbreakable/experiments/outcome_names_ordered.json", "r") as f:
        return json.load(f)


def build_column_list(
    base_columns: List[str],
    policy_names: List[str],
    uncertainty_names: List[str],
    outcome_names: List[str],
    include_policies: bool,
    include_uncertainties: bool,
) -> List[str]:
    """Build the list of columns for the output DataFrame."""
    columns = base_columns.copy()
    if include_policies:
        columns.extend(policy_names)
    if include_uncertainties:
        columns.extend(uncertainty_names)
    columns.extend(outcome_names)
    return columns


def process_spatial_unit_data(
    experiments: pd.DataFrame,
    spatial_unit_outcomes: np.ndarray,
    spatial_unit: str,
    policy_names: List[str],
    uncertainty_names: List[str],
    outcome_names: List[str],
    include_policies: bool,
    include_uncertainties: bool,
) -> List[List[Any]]:
    """Process data for a single spatial unit."""
    spatial_unit_data = []
    for idx, outcome_array in enumerate(spatial_unit_outcomes):
        row = [
            experiments["scenario"].iloc[idx],
            experiments["policy"].iloc[idx],
            spatial_unit,
            experiments["random_seed"].iloc[idx],
        ]

        if include_policies:
            row.extend(experiments[policy_names].iloc[idx].tolist())

        if include_uncertainties:
            row.extend(experiments[uncertainty_names].iloc[idx].tolist())

        row.extend(process_outcome_values(outcome_array, outcome_names))
        spatial_unit_data.append(row)

    return spatial_unit_data


def process_outcome_values(
    outcome_array: np.ndarray, outcome_names: List[str]
) -> List[Any]:
    """Process individual outcome values."""
    processed_values = []
    for value, name in zip(outcome_array, outcome_names):
        if name in ["years_in_poverty"]:
            processed_values.append(ast.literal_eval(value))
        else:
            processed_values.append(value)
    return processed_values
