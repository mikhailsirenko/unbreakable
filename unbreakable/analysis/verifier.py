import os
import pandas as pd
from tqdm import tqdm


def filter_and_save_consumption_data(country: str = 'Nigeria', return_period: int = 100, run_subset: bool = False, n_runs: int = 1, include_regions_of_interest: bool = False, regions_of_interest: list = None, save_results: bool = False):
    '''
    Filter and save the consumption recovery data for the specified country.

    Args:
        country (str): The country for which to load the data.
        return_period (int): The return period of the disaster.
        run_subset (bool): If True, only the first n_runs runs are included. If False, all runs are included.
        n_runs (int): The number of runs to include if run_subset is True.
        include_regions_of_interest (bool): If True, only data from regions are included. If False, data from all regions are included.
        regions_of_interest (list, optional): A list of regions of interest. Only used if include_regions_of_interest is True.
        save_results (bool): If True, the results are saved to a CSV file.

    Returns:
        dict: A dictionary with keys 'Conflict' and 'No Conflict', each containing a dictionary with 'region' and 'data'.

    Examples:
        # Load only 1 run
        run_subset = True
        n_runs = 1

        # Load only data from Oyo and Taraba
        include_regions_of_interest = True
        regions_of_interest = ['Oyo', 'Taraba']

        # Run the function
        results = filter_and_save_consumption_data(run_subset=run_subset, n_runs=n_runs, include_regions_of_interest=include_regions_of_interest, regions_of_interest=regions_of_interest, save_results=True)
    '''
    # Initialize the results dictionary
    results = {}

    # False - No Conflict, True - Conflict
    run_types = [False, True]

    # Loop through the two run types
    for c in run_types:
        # Set the path to the experiment data
        path = os.path.abspath(
            f'../../experiments/{country}/consumption_recovery/return_period={return_period}/conflict={c}/')

        # Get the list of folders in the path
        folders = [f for f in os.listdir(
            path) if os.path.isdir(os.path.join(path, f))]

        # Limit the number of runs if run_subset is True
        if run_subset:
            folders = folders[:n_runs]
        else:
            n_runs = len(folders)

        # Initialize the dictionary to store the data for all regions
        all_regions_data = {}

        # Loop through the folders
        for folder in tqdm(folders, desc=f"Processing {'Conflict' if c else 'No Conflict'} Runs"):
            # Get the list of files in the folder
            folder_path = os.path.join(path, folder)

            # Limit the files to the regions of interest if include_regions_of_interest is True
            if include_regions_of_interest:
                files = [f for f in os.listdir(folder_path) if f.split('.')[
                    0] in regions_of_interest]
            else:
                files = os.listdir(folder_path)

            # Check if region should be included based on the include_conflict_regions_only flag and conflict_regions list
            for file in files:
                # Get the region name from the file name
                region = file.split('.')[0]

                # Load the data from the file
                file_path = os.path.join(folder_path, file)
                data = pd.read_pickle(file_path)

                # t is the time index
                t = list(data.keys())

                # Get the consumption recovery data for each region
                all_region_hh = [data[i]['c_t'].rename(
                    i) for i in t if not data[i].empty]

                # Concatenate the data for all households in the region
                all_regions_data[region] = pd.concat(all_region_hh, axis=1)

        # Store the results in the dictionary
        results['Conflict' if c else 'No Conflict'] = all_regions_data

    # Save the results to a CSV file
    if save_results:
        results_path = os.path.abspath(
            f'../../experiments/{country}/consumption_recovery/')
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        for run_type, regions_data in results.items():
            for region, data in regions_data.items():
                data.to_csv(f'{results_path}/{run_type}_{region}_{n_runs}.csv')

    return results


def load_consumption_data(country: str = 'Nigeria') -> list:
    '''Load the prepared consumption recovery data.'''
    # List the files in the folder
    folder = f'../../experiments/{country}/consumption_recovery/'
    files = os.listdir(folder)

    # Get the regional and mean files
    regional_files = [f for f in files if 'mean' not in f]

    # Ignore files if these are folders
    regional_files = [f for f in regional_files if '.' in f]

    # Load the data from the regional files
    # Differentiate between conflict and no conflict
    conflict_regional_data = {}
    no_conflict_regional_data = {}

    for file in regional_files:
        # Get the region name from the file name
        region = file.split('_')[1]
        # Get the conflict status from the file name
        conflict_status = 'No Conflict' if 'No Conflict' in file else 'Conflict'
        # Load the data from the file
        data = pd.read_csv(folder + file, index_col=0)
        # Store the data in the appropriate dictionary
        if conflict_status == 'Conflict':
            conflict_regional_data[region] = data
        else:
            no_conflict_regional_data[region] = data

    return conflict_regional_data, no_conflict_regional_data


filter_and_save_consumption_data(save_results=True)
