from unbreakable.data.reader import *

def test_read_asset_damage():
    '''Test read_asset_damage function.'''
    all_damage = read_asset_damage('Saint Lucia')
    assert isinstance(all_damage, pd.DataFrame)
    assert all_damage.shape[0] == 9 # Saint Lucia has 9 districts

def test_read_household_survey():
    '''Test read_household_survey function.'''
    household_survey = read_household_survey('Saint Lucia')
    assert isinstance(household_survey, pd.DataFrame)
    assert household_survey['district'].unique().size == 9 # Saint Lucia has 9 districts

def test_get_expected_loss_fraction():
    '''Test get_expected_loss_fraction function.'''
    all_damage = read_asset_damage('Saint Lucia')
    districts = all_damage['district'].unique()
    for district in districts:
        return_periods = all_damage['rp'].unique()
        for return_period in return_periods:
            expected_loss_fraction = get_expected_loss_fraction(all_damage, district, return_period)
            assert isinstance(expected_loss_fraction, float)
            assert expected_loss_fraction >= 0
            assert expected_loss_fraction <= 1