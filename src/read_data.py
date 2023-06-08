import pandas as pd
import os
import numpy as np
# This file contains the Reader class, which is used to read model parameters, damage parameters and input data from files.


class Reader():
    '''Read model parameters, damage parameters and input data from files.'''

    def _read_model_parameters(self, country: str = '', state: str = '', district: str = '', filepath: str = '') -> dict:
        # !: Adjust the function, in its current version it won't work
        '''Reads model parameters from excel file and returns a dictionary

        Parameters
        ----------
        country : str
            Country name
        state : str
            State name
        district : str
            District name
        filepath : str
            Path to excel file

        Returns
        -------
        dict
            Dictionary with model parameters

        Raises
        ------
        ValueError
            If country is not supported

        '''
        if country not in ['India', 'Saint Lucia']:
            raise ValueError('Country not supported')
        else:
            if state == '':
                sheet_name = district
            else:
                sheet_name = state

            if filepath == '':
                model_parameters = pd.read_excel(
                    f'../data/internal/{country}/model_parameters.xlsx', sheet_name=sheet_name)
            else:
                model_parameters = pd.read_excel(
                    filepath, sheet_name=sheet_name)

            return model_parameters.set_index('Name').to_dict()['Value']

    def _get_file_name(self, country: str = '', state: str = '', district: str = '', scale: str = '') -> str:
        '''Get the appropriate file name based on the chosen scale.'''
        if scale == 'country':
            f = country
        elif scale == 'state':
            f = state
        elif scale == 'district':
            f = district
        return f

    def _read_asset_damage(self, column: str = 'exposed_value',  filepath: str = '') -> dict:
        '''Reads damage parameters from csv file and returns a dictionary'''
        if filepath == '':
            all_damage = pd.read_excel(
                f"../data/raw/asset_damage/{self.country}/{self.country}.xlsx", index_col=None, header=0)
        else:
            all_damage = pd.read_csv(filepath)

        # If we do not differentiate between states or districts work with the data on the country level
        # * RP - return period
        # * PML - probable maximum loss
        if self.scale == 'country':
            event_damage = all_damage.loc[all_damage['RP']
                                          == self.return_period, 'PML'].values[0]
            total_asset_stock = all_damage.loc[(
                all_damage['RP'] == self.return_period), column].values[0]

        # We have multiple states or districts, then subset it
        elif self.scale == 'state':
            event_damage = all_damage.loc[(all_damage[self.scale] == self.state) & (
                all_damage['RP'] == self.return_period), 'PML'].values[0]
            total_asset_stock = all_damage.loc[(all_damage[self.scale] == self.state) & (
                all_damage['RP'] == self.return_period), column].values[0]
        else:
            event_damage = all_damage.loc[(all_damage[self.scale] == self.district) & (
                all_damage['RP'] == self.return_period), 'PML'].values[0]
            total_asset_stock = all_damage.loc[(all_damage[self.scale] == self.district) & (
                all_damage['RP'] == self.return_period), column].values[0]

        return {'event_damage': event_damage, 'total_asset_stock': float(total_asset_stock)}

    def _read_household_data(self) -> pd.DataFrame:
        '''Reads input data from csv files and returns a dictionary'''
        household_data = pd.read_csv(self.household_data_filename)
        return household_data

    def _duplicate_households(self) -> None:
        '''Duplicates households if the number of households is less than the threshold'''
        # !: @Bramka, check out this implementation
        # Note that the previous implementation was not wrong,
        # specifically where you adjusted the weights

        if len(self.household_data) < self.min_households:
            print(
                f'Number of households = {len(self.household_data)} is less than the threshold = {self.min_households}')

            initial_total_weights = self.household_data['popwgt'].sum()

            # Save the original household id
            self.household_data['hhid_original'] = self.household_data[self.household_column_id]

            # Get random ids from the household data to be duplicated
            ids = np.random.choice(
                self.household_data.index, self.min_households - len(self.household_data), replace=False)
            n_duplicates = pd.Series(ids).value_counts() + 1
            duplicates = self.household_data.loc[ids]

            # Adjust the weights of the duplicated households
            duplicates['popwgt'] = duplicates['popwgt'] / n_duplicates

            # Adjust the weights of the original households
            self.household_data.loc[ids, 'popwgt'] = self.household_data.loc[ids, 'popwgt'] / \
                n_duplicates

            # Combine the original and duplicated households
            self.household_data = pd.concat(
                [self.household_data, duplicates], ignore_index=True)

            # Check if the total weights after duplication is equal to the initial total weights
            weights_after_duplication = self.household_data['popwgt'].sum()
            if weights_after_duplication != initial_total_weights:
                raise ValueError(
                    'Total weights after duplication is not equal to the initial total weights')

            self.household_data.reset_index(drop=True, inplace=True)
            print(
                f'Number of households after duplication: {len(self.household_data)}')
        else:
            pass

    def _calculate_average_productivity(self, print_statistics: bool = False) -> float:
        '''Calculate average productivity as aeinc \ k_house_ae'''
        average_productivity = self.household_data['aeinc'] / \
            self.household_data['k_house_ae']

        # ?: What's happening here?
        # self.average_productivity = self.average_productivity.iloc[0]
        average_productivity = np.nanmedian(average_productivity)
        if print_statistics:
            print('Average productivity of capital = ' +
                  str(np.round(average_productivity, 3)))
        return average_productivity

    def _get_optimization_results(self) -> dict:
        '''Read or create a data frame for optimization results'''
        try:
            # If we already run optimization then pick up the saves values
            # It supposed to speed the process up
            optimization_results = pd.read_csv(self.optimization_results_filename, index_col=[
                'aeexp', 'aeexp_house', 'keff', 'v', 'aesav']).sort_index()
        except:
            # Otherwise create a new data frame to store the results
            optimization_results = pd.DataFrame({'aeexp': -1, 'aeexp_house': -1, 'keff': -1,
                                                 'v': -1, 'aesav': -1, 'solution': None, 'bankrupt': None}, index=[0])
            optimization_results = optimization_results.reset_index(drop=True).set_index(
                ['aeexp', 'aeexp_house', 'keff', 'v', 'aesav'])

        return optimization_results

    def _collect_parameters(self) -> None:
        '''Collect all model parameters into `self.parameters`.'''
        self.parameters = {'poverty_line': self.poverty_line,
                           'indigence_line': self.indigence_line,
                           'saving_rate': self.saving_rate,
                           'income_and_expenditure_growth': self.income_and_expenditure_growth,
                           'poverty_bias': self.poverty_bias,
                           'discount_rate': self.discount_rate,
                           'consumption_utility': self.consumption_utility,
                           'adjust_assets_and_expenditure': self.adjust_assets_and_expenditure,
                           'n_replications': self.n_replications,
                           'optimization_timestep': self.optimization_timestep,
                           'return_period': self.return_period,
                           'policy': self.policy,
                           'event_damage': self.event_damage,
                           'total_asset_stock': self.total_asset_stock,
                           'expected_loss_fraction': self.expected_loss_fraction,
                           'average_productivity': self.average_productivity,
                           'min_households': self.min_households}

    def _prepare_data_frames(self) -> None:
        '''Prepare data frames to store simulation results'''
        self.simulation_parameters = pd.DataFrame({'poverty_bias': None}, index=[
            'replication_{}'.format(_) for _ in range(self.n_replications)])
        self.quintile_recovery_rate = pd.DataFrame({_: None for _ in range(1, self.n_replications + 1)}, index=[
            'replication_{}'.format(_) for _ in range(self.n_replications)])
        self.quintile_weeks_pov = pd.DataFrame({_: None for _ in range(1, self.n_replications + 1)}, index=[
            'replication_{}'.format(_) for _ in range(self.n_replications)])
        self.quintile_asset_loss_totval = pd.DataFrame({_: None for _ in range(1, self.n_replications + 1)}, index=[
            'replication_{}'.format(_) for _ in range(self.n_replications)])
        self.quintile_asset_loss_percap = pd.DataFrame({_: None for _ in range(1, self.n_replications + 1)}, index=[
            'replication_{}'.format(_) for _ in range(self.n_replications)])
        self.quintile_DRM_cost = pd.DataFrame({_: None for _ in range(1, self.n_replications + 1)}, index=[
            'replication_{}'.format(_) for _ in range(self.n_replications)])
        self.hh_vulnerability = pd.DataFrame(
            {'replication_{}'.format(_): None for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_transfers = pd.DataFrame(
            {'replication_{}'.format(_): None for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_savings = pd.DataFrame(
            {'replication_{}'.format(_): None for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_is_affected = pd.DataFrame(
            {'replication_{}'.format(_): None for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_asset_loss = pd.DataFrame(
            {'replication_{}'.format(_): None for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_is_impoverished = pd.DataFrame(
            {'replication_{}'.format(_): None for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_weeks_pov = pd.DataFrame(
            {'replication_{}'.format(_): None for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_reco_rate = pd.DataFrame(
            {'replication_{}'.format(_): None for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_consumption_loss = pd.DataFrame(
            {'replication_{}'.format(_): 0 for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_consumption_loss_NPV = pd.DataFrame(
            {'replication_{}'.format(_): 0 for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_welfare_loss = pd.DataFrame(
            {'replication_{}'.format(_): 0 for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_welfare_loss_updated1 = pd.DataFrame(
            {'replication_{}'.format(_): 0 for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_net_consumption_loss = pd.DataFrame(
            {'replication_{}'.format(_): 0 for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_net_consumption_loss_NPV = pd.DataFrame(
            {'replication_{}'.format(_): 0 for _ in range(self.n_replications)}, index=self.household_data.index)
        self.hh_DRM_cost = pd.DataFrame(
            {'replication_{}'.format(_): None for _ in range(self.n_replications)}, index=self.household_data.index)

    def _adjust_assets_and_expenditure(self) -> None:
        '''Adjust assets and expenditure of household to match data of asset damage file.

        There can be a mistmatch between the data in the household survey and the of the asset damage.
        The latest was created independently.'''

        # ?: Do we always have to do that?
        # If yes, remove the corresponding variable. Or else add a condition?

        # k_house_ae - effective capital stock of the household
        # aeexp - adult equivalent expenditure of a household (total)
        # aeexp_house - data['hhexp_house'] (annual rent) / data['hhsize_ae']
        included_variables = ['k_house_ae', 'aeexp', 'aeexp_house']
        total_asset_in_survey = self.household_data[[
            'popwgt', 'k_house_ae']].prod(axis=1).sum()
        scaling_factor = self.total_asset_stock / total_asset_in_survey
        self.household_data[included_variables] *= scaling_factor
        self.poverty_line *= scaling_factor
        self.indigence_line *= scaling_factor

    def _calculate_pml(self) -> None:
        '''Calculate probable maxmium loss of each household'''
        # keff - effective capital stock
        self.household_data['keff'] = self.household_data['k_house_ae'].copy()
        # pml - probable maximum loss
        # popwgt - population weight of each household
        self.pml = self.household_data[['popwgt', 'keff']].prod(
            axis=1).sum() * self.expected_loss_fraction
        print('Probable maximum loss (total) : ', '{:,}'.format(self.pml))

    def _print_parameters(self) -> None:
        '''Print all model parameters.'''
        print('Model parameters:')
        for key, value in self.parameters.items():
            print(f'{key}: {value}')
