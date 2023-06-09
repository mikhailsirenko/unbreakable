import pandas as pd
import os
import numpy as np
import json
# This file contains the Reader class, which is used to read model parameters, damage parameters and input data from files.


class Reader():
    '''Read model parameters, damage parameters and input data from files.'''

    def _read_function_parameters(self) -> None:
        '''Read function parameters from json file and store them in a dictionary'''
        with open(f'../data/internal/{self.country}/function_parameters.json') as json_file:
            self.function_parameters = json.load(json_file)

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

    def _prepare_asset_damage_data(self) -> None:
        '''Prepares asset damage data for the Saint Lucia case study'''
        if self.country == 'Saint Lucia':
            if self.scale == 'district':
                # Load raw data
                df = pd.read_excel(
                    '../data/raw/asset_damage/Saint Lucia/St Lucia 2015 exposure summary.xlsx', sheet_name='total by parish', skiprows=1)
                # Remove redundant columns
                df.drop(df.columns[0], axis=1, inplace=True)
                # Even though the data is by parish, let's call the corresponding column district
                df.rename(columns={'Unnamed: 1': 'district'}, inplace=True)
                # !: Check whether rp is = 100 given the data
                df['rp'] = 100
                df.rename(
                    columns={'Combined Total': 'exposed_value'}, inplace=True)

                # !: Replace with the real data
                # Let's assume that PML is equal to AAL % by district * by the PML for the whole country
                # These values are from PML Results 19022016 SaintLucia FinalSummary2.xlsx
                total_pml = {10: 351733.75, 50: 23523224.51, 100: 59802419.04,
                             250: 147799213.30, 500: 248310895.20, 1000: 377593847.00}
                print('Total PML: ', '{:,}'.format(round(total_pml[self.return_period])))
                aal = pd.read_excel(
                    '../data/processed/asset_damage/Saint Lucia/AAL Results 19022016 StLucia FinalSummary2 adjusted.xlsx', sheet_name='AAL St. Lucia Province')
                aal.set_index('Name', inplace=True)
                aal = aal[['AAL as % of Total AAL']]
                aal.columns = ['pml']
                aal = aal[aal.index.notnull()]
                pml = aal.multiply(total_pml[self.return_period])
                df = pd.merge(df, pml, left_on='district', right_index=True)
                df.to_excel(
                    f'../data/processed/asset_damage/{self.country}/{self.country}.xlsx', index=False)
            else:
                raise ValueError(
                    'Only district scale is supported for Saint Lucia')
        else:
            pass

    def _read_asset_damage(self, column: str = 'exposed_value',  filepath: str = '') -> dict:
        '''Reads damage parameters from csv file and returns a dictionary'''
        self._prepare_asset_damage_data()
        if filepath == '':
            all_damage = pd.read_excel(
                f"../data/processed/asset_damage/{self.country}/{self.country}.xlsx", index_col=None, header=0)
            # f"../data/raw/asset_damage/{self.country}/{self.country}.xlsx", index_col=None, header=0)
        else:
            all_damage = pd.read_csv(filepath)

        # If we do not differentiate between states or districts work with the data on the country level
        # * rp - return period
        # * pml - probable maximum loss
        if self.scale == 'country':
            event_damage = all_damage.loc[all_damage['rp']
                                          == self.return_period, 'pml'].values[0]
            total_asset_stock = all_damage.loc[(
                all_damage['rp'] == self.return_period), column].values[0]

        # We have multiple states or districts, then subset it
        elif self.scale == 'state':
            event_damage = all_damage.loc[(all_damage[self.scale] == self.state) & (
                all_damage['rp'] == self.return_period), 'pml'].values[0]
            total_asset_stock = all_damage.loc[(all_damage[self.scale] == self.state) & (
                all_damage['rp'] == self.return_period), column].values[0]
        else:
            event_damage = all_damage.loc[(all_damage[self.scale] == self.district) & (
                all_damage['rp'] == self.return_period), 'pml'].values[0] # PML
            total_asset_stock = all_damage.loc[(all_damage[self.scale] == self.district) & (
                all_damage['rp'] == self.return_period), column].values[0] # Exposed value

        return {'event_damage': event_damage, 'total_asset_stock': float(total_asset_stock)}

    def _duplicate_households(self) -> None:
        '''Duplicates households if the number of households is less than the threshold'''
        # TODO: Make sure that the weights redistribution is correct
        # !: @Bramka, check out this implementation
        # Note that the previous implementation was not wrong,
        # specifically where you adjusted the weights

        if len(self.households) < self.min_households:
            print(
                f'Number of households = {len(self.households)} is less than the threshold = {self.min_households}')

            initial_total_weights = self.households['popwgt'].sum()

            # Save the original household id
            self.households['hhid_original'] = self.households[self.household_column_id]

            # Get random ids from the household data to be duplicated
            ids = np.random.choice(
                self.households.index, self.min_households - len(self.households), replace=False)
            n_duplicates = pd.Series(ids).value_counts() + 1
            duplicates = self.households.loc[ids]

            # Adjust the weights of the duplicated households
            duplicates['popwgt'] = duplicates['popwgt'] / n_duplicates

            # Adjust the weights of the original households
            self.households.loc[ids, 'popwgt'] = self.households.loc[ids, 'popwgt'] / \
                n_duplicates

            # Combine the original and duplicated households
            self.households = pd.concat(
                [self.households, duplicates], ignore_index=True)

            # Check if the total weights after duplication is equal to the initial total weights
            # TODO: Allow for a small difference
            weights_after_duplication = self.households['popwgt'].sum()
            if weights_after_duplication != initial_total_weights:
                raise ValueError(
                    'Total weights after duplication is not equal to the initial total weights')

            self.households.reset_index(drop=True, inplace=True)
            print(
                f'Number of households after duplication: {len(self.households)}')
        else:
            pass

    def _calculate_average_productivity(self, print_statistics: bool = False) -> float:
        '''Calculate average productivity as aeinc \ k_house_ae'''
        average_productivity = self.households['aeinc'] / \
            self.households['k_house_ae']

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

    def _adjust_assets_and_expenditure(self) -> None:
        '''Adjust assets and expenditure of household to match data of asset damage file.

        There can be a mismatch between the data in the household survey and the of the asset damage.
        The latest was created independently.'''

        # ?: Do we always have to do that?
        # If yes, remove the corresponding variable. Or else add a condition?

        # k_house_ae - effective capital stock of the household
        # aeexp - adult equivalent expenditure of a household (total)
        # aeexp_house - data['hhexp_house'] (annual rent) / data['hhsize_ae']
        included_variables = ['k_house_ae', 'aeexp', 'aeexp_house']

        # Save the initial values
        self.households['k_house_ae_original'] = self.households['k_house_ae']
        self.households['aeexp_original'] = self.households['aeexp']
        self.households['aeexp_house_original'] = self.households['aeexp_house']

        total_asset_in_survey = self.households[[
            'popwgt', 'k_house_ae']].prod(axis=1).sum()
        scaling_factor = self.total_asset_stock / total_asset_in_survey
        self.households[included_variables] *= scaling_factor
        self.poverty_line *= scaling_factor
        self.indigence_line *= scaling_factor
        self.households['poverty_line_adjusted'] = self.poverty_line
        self.households['indigence_line_adjusted'] = self.indigence_line
        
        print()

    def _calculate_pml(self) -> None:
        '''Calculate probable maxmium loss of each household'''
        # keff - effective capital stock
        self.households['keff'] = self.households['k_house_ae'].copy()
        # pml - probable maximum loss
        # popwgt - population weight of each household
        self.pml = self.households[['popwgt', 'keff']].prod(axis=1).sum() * self.expected_loss_fraction
        self.households['pml'] = self.pml
        print('Probable maximum loss (total) : ',
              '{:,}'.format(round(self.pml)))

    def _print_parameters(self) -> None:
        '''Print all model parameters.'''
        print('Model parameters:')
        for key, value in self.parameters.items():
            try:
                print(f'{key}: {round(value, 2)}')
            except:
                print(f'{key}: {value}')
