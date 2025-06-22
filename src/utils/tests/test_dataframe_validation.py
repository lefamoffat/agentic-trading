import numpy as np
import pandas as pd
import pytest

from src.utils.validation.dataframe import calculate_data_quality_score

@pytest.mark.unit
class TestDataQualityScore:
    def test_perfect_quality_data(self):
        data = pd.DataFrame({
            'high': [10, 20, 30],
            'low': [5, 15, 25],
            'volume': [1000, 1000, 1000]
        })
        assert calculate_data_quality_score(data) == 1.0

    def test_empty_dataframe(self):
        data = pd.DataFrame()
        assert calculate_data_quality_score(data) == 0.0

    def test_completeness_score(self):
        data = pd.DataFrame({
            'high': [10, np.nan, 30],
            'low': [5, 15, 25],
            'volume': [1000, 1000, np.nan]
        })
        expected = (7/9)*0.4 + 1.0*0.4 + 1.0*0.2
        assert calculate_data_quality_score(data) == pytest.approx(expected)

    def test_consistency_score(self):
        data = pd.DataFrame({
            'high': [10, 15, 25],
            'low': [5, 20, 20],
            'volume': [1000, 1000, 1000]
        })
        expected = 1.0*0.4 + (2/3)*0.4 + 1.0*0.2
        assert calculate_data_quality_score(data) == pytest.approx(expected)

    def test_volume_health_score(self):
        data = pd.DataFrame({
            'high': np.arange(10,20),
            'low' : np.arange(0,10),
            'volume': [1000]*8 + [0]*2
        })
        assert calculate_data_quality_score(data) == pytest.approx(0.9)
        data['volume'] = [1000]*7 + [0]*3
        expected = 1.0*0.4 + 1.0*0.4 + 0.0*0.2
        assert calculate_data_quality_score(data) == pytest.approx(expected)

    def test_combined_quality_issues(self):
        data = pd.DataFrame({
            'high': [10,20,30,40,50],
            'low' : [5,25,25,35,45],
            'volume': [100,200,0,np.nan,500]
        })
        expected = 0.7433333333333334
        assert calculate_data_quality_score(data) == pytest.approx(expected)
