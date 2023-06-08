
# This file contains the Writer class, which is used to write the results of the simulation into a set of csv files.

class Writer():
    '''Write the results of the simulation into a set of csv files'''

    def _write_event_results(self, current_replication: int) -> None:
        '''Write the results of the event to the results dataframes'''
        data = self.household_data.copy().reset_index().set_index('quintile')
        self.quintile_asset_loss_totval.loc[current_replication, :] = data.loc[data['affected'] == True, [
            'popwgt', 'keff', 'v']].prod(axis=1).groupby(level='quintile').sum()
        self.quintile_asset_loss_percap.loc[current_replication, :] = data.loc[data['affected'] == True, ['popwgt', 'keff', 'v']].prod(
            axis=1).groupby(level='quintile').sum() / data.loc[data['affected'] == True, 'popwgt'].groupby(level='quintile').sum()
        self.quintile_DRM_cost.loc[current_replication, :] = data[['popwgt', 'DRM_cost']].prod(
            axis=1).groupby(level='quintile').sum()

    def _write_household_results(self, current_replication: int) -> None:
        '''Write household impacts into a set of dataframes'''
        self.hh_vulnerability[current_replication] = self.affected_households['v']
        self.hh_transfers[current_replication] = self.affected_households['aesoc'].fillna(
            0)
        self.hh_savings[current_replication] = self.affected_households['aesav'].fillna(
            0)

        self.hh_is_affected[current_replication] = self.affected_households['popwgt']
        self.hh_asset_loss[current_replication] = self.affected_households[[
            'keff', 'v']].prod(axis=1)
        self.hh_is_impoverished[current_replication] = self.affected_households.loc[
            self.affected_households['weeks_pov'] > 0, 'popwgt']
        self.hh_weeks_pov[current_replication] = self.affected_households['weeks_pov']
        self.hh_reco_rate[current_replication] = self.affected_households['reco_rate']
        self.hh_consumption_loss.loc[self.household_data['affected'],
                                     current_replication] = self.affected_households['consumption_loss']
        self.hh_consumption_loss.loc[~self.household_data['affected'], current_replication] = - \
            self.household_data.loc[~self.household_data['affected'], 'DRM_cash']
        self.hh_welfare_loss.loc[self.household_data['affected'],
                                 current_replication] = self.affected_households['w_final']
        self.hh_welfare_loss.loc[~self.household_data['affected'],
                                 current_replication] = 0
        self.hh_welfare_loss_updated1.loc[self.household_data['affected'],
                                          current_replication] = self.affected_households['w_final2']
        self.hh_welfare_loss_updated1.loc[~self.household_data['affected'],
                                          current_replication] = 0
        self.hh_consumption_loss_NPV.loc[self.household_data['affected'],
                                         current_replication] = self.affected_households['consumption_loss_NPV']
        self.hh_consumption_loss_NPV.loc[~self.household_data['affected'],
                                         current_replication] = -self.household_data.loc[~self.household_data['affected'], 'DRM_cash']
        self.hh_net_consumption_loss.loc[self.household_data['affected'],
                                         current_replication] = self.affected_households['net_consumption_loss']
        self.hh_net_consumption_loss_NPV.loc[self.household_data['affected'],
                                             current_replication] = self.affected_households['net_consumption_loss_NPV']
        self.hh_DRM_cost[current_replication] = self.household_data['DRM_cost']

        self.affected_households = self.affected_households.reset_index().set_index('quintile')

        self.quintile_weeks_pov.loc[current_replication, :] = self.affected_households[['popwgt', 'weeks_pov']].prod(
            axis=1).groupby(level='quintile').sum() / self.affected_households['popwgt'].groupby(level='quintile').sum()
        self.quintile_recovery_rate.loc[current_replication, :] = self.affected_households[['popwgt', 'reco_rate']].prod(
            axis=1).groupby(level='quintile').sum() / self.affected_households['popwgt'].groupby(level='quintile').sum()

    def _save_affected_household_data(self, current_replication: int) -> None:
        '''Save the household data for the affected households into a separate csv file'''
        self.affected_households.to_csv(
            self.results_directory + f'/affected_hh_data_{current_replication}.csv')

    def _save_simulation_results(self) -> None:
        '''Save the simulation results into a set of csv files'''
        self.simulation_parameters.to_csv(
            '{}/simulation_params.csv'.format(self.results_directory))
        self.quintile_recovery_rate.to_csv(
            '{}/quintile_recovery_rate.csv'.format(self.results_directory))
        self.quintile_weeks_pov.to_csv(
            '{}/quintile_weeks_pov.csv'.format(self.results_directory))
        self.quintile_asset_loss_totval.to_csv(
            '{}/quintile_asset_loss_totval.csv'.format(self.results_directory))
        self.quintile_asset_loss_percap.to_csv(
            '{}/quintile_asset_loss_percap.csv'.format(self.results_directory))
        self.quintile_DRM_cost.to_csv(
            '{}/quintile_DRM_cost.csv'.format(self.results_directory))
        self.hh_vulnerability.to_csv(
            '{}/hh_vulnerability.csv'.format(self.results_directory))
        self.hh_transfers.to_csv(
            '{}/hh_transfers.csv'.format(self.results_directory))
        self.hh_savings.to_csv(
            '{}/hh_savings.csv'.format(self.results_directory))
        self.hh_is_affected.to_csv(
            '{}/hh_is_affected.csv'.format(self.results_directory))
        self.hh_asset_loss.to_csv(
            '{}/hh_asset_loss.csv'.format(self.results_directory))
        self.hh_is_impoverished.to_csv(
            '{}/hh_is_impoverished.csv'.format(self.results_directory))
        self.hh_weeks_pov.to_csv(
            '{}/hh_weeks_pov.csv'.format(self.results_directory))
        self.hh_reco_rate.to_csv(
            '{}/hh_reco_rate.csv'.format(self.results_directory))
        self.hh_consumption_loss.to_csv(
            '{}/hh_consumption_loss.csv'.format(self.results_directory))
        self.hh_consumption_loss_NPV.to_csv(
            '{}/hh_consumption_loss_NPV.csv'.format(self.results_directory))
        self.hh_welfare_loss.to_csv(
            '{}/hh_welfare_loss.csv'.format(self.results_directory))
        self.hh_welfare_loss_updated1.to_csv(
            '{}/hh_welfare_loss_updated1.csv'.format(self.results_directory))
        self.hh_net_consumption_loss.to_csv(
            '{}/hh_net_consumption_loss.csv'.format(self.results_directory))
        self.hh_net_consumption_loss_NPV.to_csv(
            '{}/hh_net_consumption_loss_NPV.csv'.format(self.results_directory))
        self.hh_DRM_cost.to_csv(
            '{}/hh_DRM_cost.csv'.format(self.results_directory))