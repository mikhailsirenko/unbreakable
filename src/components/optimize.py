import numpy as np

# This file contains the Optimizer class, which is used to optimize the recovery rate for each affected household.


class Optimizer():
    '''Optimize the recovery rate for each affected household.'''

    def _run_optimization(self) -> None:
        print('Running optimization...')
        # TODO: Add docstring
        # ? What does it mean?
        # Set KEFF for renters to zero, though they still lose housing service
        self.affected_households.loc[self.affected_households['own_rent']
                                     == 'rent', 'keff'] = 0
        self.affected_households['reco_rate'] = self.affected_households['v'].apply(
            lambda x: self._optimize_reco2(x))
        self.optimization_results = self.optimization_results.sort_index()
        print('Optimization done!')

    def _optimize_reco2(self, x) -> float:
        # TODO: Add docstring
        # TODO: Do not hard code the parameters here. Move it to a config file.
        try:
            # Look for the existing solution
            solution = self.optimization_results.loc[(
                0, 0, 0, round(x, 3), round(self.average_productivity, 3)), 'solution']
            return solution
        except:
            # No existing solution found, so we need to optimize
            t_max_linspace = 10
            nsteps_linspace = 52*t_max_linspace
            dt = t_max_linspace / nsteps_linspace

            _lambda = 0
            opt_step = self.optimization_timestep
            last_dwdlambda = 0

            while True:

                dwdlambda = 0

                for _t in np.linspace(0, t_max_linspace, nsteps_linspace):

                    part1 = self.average_productivity - \
                        (self.average_productivity+_lambda)*x*np.e**(-_lambda*_t)
                    part1 = part1**(-self.consumption_utility)

                    part2 = _t * (self.average_productivity+_lambda) - 1

                    part3 = np.e**(-_t*(self.discount_rate+_lambda))

                    dwdlambda += part1 * part2 * part3 * dt

                # print([_lambda, dwdlambda])

                if last_dwdlambda < 0 and dwdlambda > 0:
                    # record
                    self.optimization_results.loc[(0, 0, 0, round(x, 3), round(self.average_productivity, 3)), [
                        'solution', 'bankrupt']] = [_lambda, False]
                    self.optimization_results = self.optimization_results.sort_index()
                    # return
                    return _lambda
                elif last_dwdlambda > 0 and dwdlambda < 0:
                    # record
                    self.optimization_results.loc[(0, 0, 0, round(x, 3), round(self.average_productivity, 3)), [
                        'solution', 'bankrupt']] = [_lambda, False]
                    self.optimization_results = self.optimization_results.sort_index()
                    # return
                    return _lambda
                elif _lambda > 10:
                    # record
                    self.optimization_results.loc[(0, 0, 0, round(x, 3), round(self.average_productivity, 3)), [
                        'solution', 'bankrupt']] = [_lambda, False]
                    self.optimization_results = self.optimization_results.sort_index()
                    # return
                    return _lambda
                else:
                    last_dwdlambda = dwdlambda
                _lambda += opt_step

    def _integrate_wellbeing(self) -> None:
        # TODO: Add docstring
        # TODO: Document the assumptions of this function
        # TODO: Do not hard code the parameters here. Move it to a config file.
        # TODO: Extensive testing

        # self.affected_households['w_init'] = (self.affected_households['keff']*self.average_productivity+df_aff['aesoc'])**(1-consumption_utility)/(discount_rate*(1-consumption_utility))
        self.affected_households[['consumption_loss', 'consumption_loss_NPV',
                                  'net_consumption_loss', 'net_consumption_loss_NPV']] = [0., 0., 0., 0.]

        x_max = 10
        tstep_n = 52*x_max
        dt = x_max/tstep_n

        self.affected_households[['c_t', 'w_final',
                                  'weeks_pov', 'w_final2']] = [0, 0, 0, 0]
        for _t in np.linspace(0, x_max, tstep_n):
            # ? What does it mean?
            # TODO: dynamic policy e.g. cash transfer at time t

            # print(poverty_line, indigence_line)

            gfac = (1+self.income_and_expenditure_growth)**_t

            # consumption at this timestep
            self.affected_households['c_t'] = (self.affected_households['aeexp']*gfac
                                               + np.e**(-self.affected_households['reco_rate']*_t) * (self.affected_households['aesav']*self.affected_households['reco_rate'] - self.affected_households['v']*gfac*(self.affected_households['aeexp_house'] + self.affected_households[['keff', 'reco_rate']].prod(axis=1)) - (1-self.affected_households['delta_tax_safety'])*self.average_productivity*self.affected_households['keff']*self.affected_households['v']))
            self.affected_households['c_t_unaffected'] = self.affected_households['aeexp']*gfac
            # consumption cannot be lower than 0
            self.affected_households.loc[self.affected_households['c_t']
                                         < 1, 'c_t'] = 1
            # consumption after hit by disaster should be lower than or equal to consumption before hit by disaster
            self.affected_households.loc[self.affected_households['c_t'] > self.affected_households['c_t_unaffected'],
                                         'c_t'] = self.affected_households.loc[self.affected_households['c_t'] > self.affected_households['c_t_unaffected'], 'c_t_unaffected']

            # total (integrated) consumption loss
            self.affected_households['consumption_loss'] += dt * \
                (self.affected_households['c_t_unaffected'] -
                 self.affected_households['c_t'])
            self.affected_households['consumption_loss_NPV'] += dt * \
                (self.affected_households['c_t_unaffected'] -
                 self.affected_households['c_t'])*np.e**(-self.discount_rate*_t)

            self.affected_households['net_consumption_loss'] += dt * \
                np.e**(-self.affected_households['reco_rate']*_t) * \
                self.affected_households['v']*gfac * \
                self.affected_households['aeexp_house']
            self.affected_households['net_consumption_loss_NPV'] += dt * \
                np.e**(-self.affected_households['reco_rate']*_t) * self.affected_households['v']*gfac * \
                self.affected_households['aeexp_house'] * \
                np.e**(-self.discount_rate*_t)
            # weeks in poverty
            self.affected_households.loc[self.affected_households['c_t']
                                         < self.poverty_line, 'weeks_pov'] += 1

            # integrated wellbeing
            self.affected_households['w_final'] += dt*(self.affected_households['c_t'])**(1-self.consumption_utility) * \
                np.e**(-self.discount_rate*_t)/(1-self.consumption_utility)

            # w_final2 version 01 - there is a bug
            # self.affected_households['w_final2'] += self.affected_households['c_t_unaffected']**(1-consumption_utility)/(1-consumption_utility)*dt*((1-(self.affected_households['consumption_loss']/self.affected_households['c_t_unaffected'])*np.e**(-self.affected_households['reco_rate']*_t))**(1-consumption_utility)-1)*np.e**(-self.discount_rate*_t)

            # w_final2 version 02
            self.affected_households['w_final2'] += self.affected_households['c_t_unaffected']**(1-self.consumption_utility)/(1-self.consumption_utility)*dt*(
                (1-((self.affected_households['c_t_unaffected'] - self.affected_households['c_t'])/self.affected_households['c_t_unaffected'])*np.e**(-self.affected_households['reco_rate']*_t))**(1-self.consumption_utility)-1)*np.e**(-self.discount_rate*_t)