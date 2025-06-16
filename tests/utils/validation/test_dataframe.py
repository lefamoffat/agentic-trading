import pytest
import pandas as pd
import numpy as np

from src.utils.validation.dataframe import calculate_data_quality_score

@pytest.mark.unit
class TestDataQualityScore:
    """Test suite for the calculate_data_quality_score function."""

    def test_perfect_quality_data(self):
        """Test with a perfect dataframe, expecting a score of 1.0."""
        data = pd.DataFrame({
            'high': [10, 20, 30],
            'low': [5, 15, 25],
            'volume': [1000, 1000, 1000]
        })
        assert calculate_data_quality_score(data) == 1.0

    def test_empty_dataframe(self):
        """Test with an empty dataframe, expecting a score of 0.0."""
        data = pd.DataFrame()
        assert calculate_data_quality_score(data) == 0.0

    def test_completeness_score(self):
        """Test the completeness component of the score."""
        # 2 missing cells out of 9 total cells (completeness = 7/9 = 0.777)
        # Score should be 0.777 * 0.4 (completeness weight) + 1.0 * 0.4 (consistency) + 1.0 * 0.2 (volume)
        data = pd.DataFrame({
            'high': [10, np.nan, 30],
            'low': [5, 15, 25],
            'volume': [1000, 1000, np.nan]
        })
        expected_score = (7/9) * 0.4 + 1.0 * 0.4 + 1.0 * 0.2
        assert calculate_data_quality_score(data) == pytest.approx(expected_score)

    def test_consistency_score(self):
        """Test the OHLC consistency component of the score."""
        # 1 inconsistent row (high < low) out of 3 total rows (consistency = 2/3 = 0.666)
        # Score should be 1.0 * 0.4 + (2/3) * 0.4 + 1.0 * 0.2
        data = pd.DataFrame({
            'high': [10, 15, 25],
            'low': [5, 20, 20], # This row is inconsistent
            'volume': [1000, 1000, 1000]
        })
        expected_score = 1.0 * 0.4 + (2/3) * 0.4 + 1.0 * 0.2
        assert calculate_data_quality_score(data) == pytest.approx(expected_score)
        
    def test_volume_health_score(self):
        """Test the volume health component of the score."""
        # 2 zero-volume rows out of 10 (20% ratio).
        # Penalty ratio = (0.2 - 0.1) / 0.2 = 0.5. Health score = 1.0 - 0.5 = 0.5
        # Final score = 1.0 * 0.4 + 1.0 * 0.4 + 0.5 * 0.2 = 0.9
        data = pd.DataFrame({
            'high': np.arange(10, 20),
            'low': np.arange(0, 10),
            'volume': [1000]*8 + [0]*2
        })
        assert calculate_data_quality_score(data) == pytest.approx(0.9)
        
        # 3 zero-volume rows out of 10 (30% ratio), should have 0 health score
        data['volume'] = [1000]*7 + [0]*3
        expected_score = 1.0 * 0.4 + 1.0 * 0.4 + 0.0 * 0.2
        assert calculate_data_quality_score(data) == pytest.approx(expected_score)

    def test_combined_quality_issues(self):
        """Test with a dataframe that has multiple quality problems."""
        # 1/10 missing cells (completeness = 0.9)
        # 1/5 inconsistent rows (consistency = 0.8)
        # 1/5 zero-volume rows (20% ratio -> volume health = 0.5)
        data = pd.DataFrame({
            'high': [10, 20, 30, 40, 50],
            'low': [5, 25, 25, 35, 45], # inconsistent
            'volume': [100, 200, 0, np.nan, 500] # nan and zero
        })
        
        completeness = 14 / 15
        consistency = 4 / 5
        # zero volume ratio = 1/5 = 0.2. penalty = (0.2-0.1)/0.2=0.5. score = 0.5
        volume_health = 0.5
        
        expected_score = (completeness * 0.4) + (consistency * 0.4) + (volume_health * 0.2)
        assert calculate_data_quality_score(data) == pytest.approx(expected_score) 