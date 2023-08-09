from unbreakable.data.reader import *
from unbreakable.modules.households import *

def test_median_productivity():
    household_survey = read_household_survey('Saint Lucia')
    for district in household_survey['district'].unique():
        households = household_survey[household_survey['district'] == district].copy()
        households = calculate_median_productivity(households)
        median_productivity = households['median_productivity'].iloc[0]
        assert median_productivity > 0