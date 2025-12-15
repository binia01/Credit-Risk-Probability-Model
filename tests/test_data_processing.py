import pytest
import pandas as pd
import numpy as np
from src.data_processing import TimeFeatureExtractor, CustomerAggregator

# Sample Data Fixture
@pytest.fixture
def sample_transaction_data():
    """Creates a small mock dataset for testing."""
    data = {
        'AccountId': ['Acc1', 'Acc1', 'Acc2'],
        'TransactionStartTime': ['2025-01-01T10:00:00Z', '2025-01-02T12:00:00Z', '2025-01-03T10:00:00Z'],
        'Value': [100.0, 200.0, 50.0],
        'Amount': [100.0, 200.0, 50.0],
        'ProductCategory': ['Airtime', 'Data', 'Airtime'],
        'ChannelId': ['Web', 'Web', 'Android'],
        'PricingStrategy': ['A', 'A', 'B'],
        'FraudResult': [0, 0, 0]
    }
    return pd.DataFrame(data)

def test_time_feature_extractor(sample_transaction_data):
    """
    Test 1: Check if TimeFeatureExtractor correctly creates Hour, Day, Month, Year columns.
    """
    extractor = TimeFeatureExtractor()
    df_transformed = extractor.transform(sample_transaction_data)
    
    expected_cols = ['Transaction_Hour', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year']
    
    # Check if columns exist
    for col in expected_cols:
        assert col in df_transformed.columns, f"Column {col} missing after transformation"
    
    # Check logic (e.g., first row hour should be 10)
    assert df_transformed.iloc[0]['Transaction_Hour'] == 10
    assert df_transformed.iloc[0]['Transaction_Year'] == 2025

def test_customer_aggregator_output_shape(sample_transaction_data):
    """
    Test 2: Check if Aggregator reduces rows to unique AccountIds (N transactions -> N customers).
    """
    # Pre-process time first (Aggregator needs datetime)
    sample_transaction_data['TransactionStartTime'] = pd.to_datetime(sample_transaction_data['TransactionStartTime'])
    
    aggregator = CustomerAggregator()
    df_agg = aggregator.transform(sample_transaction_data)
    
    # We have 3 transactions but only 2 unique Accounts (Acc1, Acc2)
    assert len(df_agg) == 2
    assert 'Recency' in df_agg.columns
    assert 'Monetary_Total' in df_agg.columns
    
    # Check Math: Acc1 spent 100 + 200 = 300
    acc1_data = df_agg[df_agg['AccountId'] == 'Acc1'].iloc[0]
    assert acc1_data['Monetary_Total'] == 300.0

def test_missing_column_error(sample_transaction_data):
    """
    Test 3: Ensure pipeline raises error if critical columns are missing.
    """
    bad_data = sample_transaction_data.drop(columns=['TransactionStartTime'])
    extractor = TimeFeatureExtractor()
    
    with pytest.raises(KeyError):
        extractor.transform(bad_data)