import pandas as pd
import os
import numpy as np
import json


class Reader():
    '''Reader class for preparing data for the model.'''

    def _read_asset_damage(self) -> None:
        '''Read asset damage for all districts from a XLSX file and load it into the memory.'''
        if self.country == 'Saint Lucia':
            self.all_damage = pd.read_excel(
                f"../data/processed/asset_damage/{self.country}/{self.country}.xlsx", index_col=None, header=0)
        else:
            raise ValueError('Only `Saint Lucia` is supported.')

    def _get_asset_damage(self, district) -> None:
        '''Get asset damage for a specific district.'''
        if self.scale == 'district':
            event_damage = self.all_damage.loc[(self.all_damage[self.scale] == district) & (
                self.all_damage['rp'] == self.return_period), 'pml'].values[0]  # PML
            total_asset_stock = self.all_damage.loc[(self.all_damage[self.scale] == district) & (
                self.all_damage['rp'] == self.return_period), 'exposed_value'].values[0]  # Exposed value

        else:
            raise ValueError(
                'Only `district` scale is supported.')

        self.event_damage = event_damage
        self.total_asset_stock = total_asset_stock
        self.expected_loss_fraction = self.event_damage / self.total_asset_stock

        if self.expected_loss_fraction > 1:
            raise ValueError(
                'Expected loss fraction is greater than 1. Check the data.')

        if self.print_statistics:
            print('Event damage = ' + str('{:,}'.format(round(event_damage))))
            print('Total asset stock = ' +
                    str('{:,}'.format(round(total_asset_stock))))
            print('Expected loss fraction = ' +
                    str(np.round(self.expected_loss_fraction, 3)))

    def _read_household_survey(self) -> None:
        '''Reads household survey from a CSV file.'''
        self.household_survey = pd.read_csv(
            f"../data/processed/household_survey/{self.country}/{self.country}.csv")

    def _prepare_households(self, district) -> None:
        '''Prepare district-level household data.
        
        Subset the household survey data for a specific district, 
        calculate average productivity and 
        adjust assets and expenditure for it.

        Args:
            district (str): District name.
        '''
        self.households = self.household_survey[self.household_survey['district'] == district].copy(
        )
        self._calculate_average_productivity()
        self._adjust_assets_and_expenditure()

    def _duplicate_households(self) -> None:
        '''Duplicates households if the number of households is less than the threshold.
        
        Raises:
            ValueError: If the total weights after duplication is not equal to the initial total weights.
        '''

        if len(self.household_survey) < self.min_households:
            print(
                f'Number of households = {len(self.household_survey)} is less than the threshold = {self.min_households}')

            initial_total_weights = self.household_survey['popwgt'].sum()

            # Save the original household id
            self.household_survey['hhid_original'] = self.household_survey[[
                'hhid']]

            # Get random ids from the household data to be duplicated
            ids = np.random.choice(
                self.household_survey.index, self.min_households - len(self.household_survey), replace=True)
            n_duplicates = pd.Series(ids).value_counts() + 1
            duplicates = self.household_survey.loc[ids]

            # Adjust the weights of the duplicated households
            duplicates['popwgt'] = duplicates['popwgt'] / n_duplicates

            # Adjust the weights of the original households
            self.household_survey.loc[ids, 'popwgt'] = self.household_survey.loc[ids, 'popwgt'] / \
                n_duplicates

            # Combine the original and duplicated households
            self.household_survey = pd.concat(
                [self.household_survey, duplicates], ignore_index=True)

            # Check if the total weights after duplication is equal to the initial total weights
            # TODO: Allow for a small difference
            weights_after_duplication = self.household_survey['popwgt'].sum()
            if weights_after_duplication != initial_total_weights:
                raise ValueError(
                    'Total weights after duplication is not equal to the initial total weights')

            self.household_survey.reset_index(drop=True, inplace=True)
            print(
                f'Number of households after duplication: {len(self.household_survey)}')
        else:
            pass

    def _calculate_average_productivity(self) -> float:
        '''Calculate average productivity as aeinc \ k_house_ae.
        
        Args:
            print_statistics (bool, optional): Whether to print the average productivity. Defaults to False.

        Returns:
            float: Average productivity.
        '''
        # DEFF: aeinc - some type of income
        average_productivity = self.households['aeinc'] / \
            self.households['k_house_ae']

        # ?: What's happening here?
        # self.average_productivity = self.average_productivity.iloc[0]
        average_productivity = np.nanmedian(average_productivity)
        if self.print_statistics:
            print('Average productivity of capital = ' +
                  str(np.round(average_productivity, 3)))
        self.average_productivity = average_productivity

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
        self.total_asset_in_survey = total_asset_in_survey
        scaling_factor = self.total_asset_stock / total_asset_in_survey
        self.households[included_variables] *= scaling_factor
        self.households['poverty_line_adjusted'] = self.poverty_line * scaling_factor
        self.households['indigence_line_adjusted'] = self.indigence_line * scaling_factor

        if self.print_statistics:
            print('Total asset in survey =',
                  '{:,}'.format(round(total_asset_in_survey)))
            print('Total asset in asset damage file =',
                  '{:,}'.format(round(self.total_asset_stock)))
            print('Scaling factor =', round(scaling_factor, 3))


    def _calculate_pml(self) -> None:
        '''Calculate probable maxmium loss as a product of population weight, effective capital stock and expected loss fraction.'''
        # DEF: keff - effective capital stock
        # DEF: pml - probable maximum loss
        # DEF: popwgt - population weight of each household
        self.households['keff'] = self.households['k_house_ae'].copy()
        self.pml = self.households[['popwgt', 'keff']].prod(
            axis=1).sum() * self.expected_loss_fraction
        
        self.households['pml'] = self.pml
        if self.print_statistics:
            print('Probable maximum loss (total) : ',
                  '{:,}'.format(round(self.pml)))
            
    def _read_optimization_results(self) -> dict:
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
        self.optimization_results = optimization_results
