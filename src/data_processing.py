import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Custom Transformations
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts Hour, Day, Month, and Year from TransactionStartTime.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'TransactionStartTime' in X.columns:
            X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
            X['Transaction_Hour'] = X['TransactionStartTime'].dt.hour
            X['Transaction_Day'] = X['TransactionStartTime'].dt.day
            X['Transaction_Month'] = X['TransactionStartTime'].dt.month
            X['Transaction_Year'] = X['TransactionStartTime'].dt.year
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Applies One-Hot Encoding to categorical variables.
    """
    def __init__(self, columns=None):
        self.columns = columns
        self.encoder = None
        self.feature_names = None

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object']).columns.tolist()
            self.columns = [c for c in self.columns if 'Id' not in c and c != 'TransactionStartTime']
        
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(X[self.columns])
        self.feature_names = self.encoder.get_feature_names_out(self.columns)
        return self

    def transform(self, X):
        X = X.copy()
        encoded_data = self.encoder.transform(X[self.columns])
        encoded_df = pd.DataFrame(encoded_data, columns=self.feature_names, index=X.index)
        
        X = X.drop(columns=self.columns)
        X = pd.concat([X, encoded_df], axis=1)
        return X

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transactions by AccountId to create customer-level features.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_rules = {
            'Value': ['sum', 'mean', 'count', 'std']
        }
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Value', 'Amount', 'Transaction_Hour', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year', 'FraudResult']:
                 if col != 'BatchId' and col != 'AccountId':
                    agg_rules[col] = 'sum'
                    
        if 'AccountId' not in X.columns:
            raise ValueError("AccountId column missing for aggregation")

        df_agg = X.groupby('AccountId').agg(agg_rules)
        df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
        
        df_agg.rename(columns={
            'Value_sum': 'Total_Transaction_Amount',
            'Value_mean': 'Average_Transaction_Amount',
            'Value_count': 'Transaction_Count',
            'Value_std': 'Std_Transaction_Amount'
        }, inplace=True)
        
        df_agg.reset_index(inplace=True)
        return df_agg

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputer = None
        self.cols_to_impute = None

    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.cols_to_impute = X.select_dtypes(include=[np.number]).columns
        self.imputer.fit(X[self.cols_to_impute])
        return self

    def transform(self, X):
        X = X.copy()
        imputed_data = self.imputer.transform(X[self.cols_to_impute])
        X[self.cols_to_impute] = imputed_data
        return X

class CustomWoETransformer(BaseEstimator, TransformerMixin):
    """
    Custom implementation of Weight of Evidence (WoE) transformation.
    """
    def __init__(self, target_col='RiskLabel', n_bins=5):
        self.target_col = target_col
        self.n_bins = n_bins
        self.woe_maps = {}
        self.bin_edges = {}
        self.feature_names = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("WoE fitting requires a target variable (y).")
        
        self.feature_names = X.columns
        df = X.copy()
        df['target'] = y.values
        
        total_good = df['target'].sum()
        total_bad = df['target'].count() - total_good
        if total_good == 0: total_good = 1
        if total_bad == 0: total_bad = 1

        for col in self.feature_names:
            try:
                if df[col].nunique() > self.n_bins:
                    _, bins = pd.qcut(df[col], q=self.n_bins, retbins=True, duplicates='drop')
                    self.bin_edges[col] = bins
                    df[f'{col}_bins'] = pd.cut(df[col], bins=bins, include_lowest=True)
                else:
                    self.bin_edges[col] = None
                    df[f'{col}_bins'] = df[col]

                grouped = df.groupby(f'{col}_bins', observed=False)['target'].agg(['count', 'sum'])
                grouped['good'] = grouped['sum']
                grouped['bad'] = grouped['count'] - grouped['sum']
                
                grouped['dist_good'] = (grouped['good'] + 0.5) / total_good
                grouped['dist_bad'] = (grouped['bad'] + 0.5) / total_bad
                
                grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
                self.woe_maps[col] = grouped['woe'].to_dict()
                
            except Exception:
                self.woe_maps[col] = {}

        return self

    def transform(self, X):
        X_woe = X.copy()
        for col in self.feature_names:
            if col in self.woe_maps and self.woe_maps[col]:
                # 1. Apply Binning (creates Categorical Series)
                if self.bin_edges.get(col) is not None:
                    binned_series = pd.cut(X_woe[col], bins=self.bin_edges[col], include_lowest=True)
                else:
                    binned_series = X_woe[col]
                
                # 2. Map to WoE values (creates Object/Float Series with NaNs)
                # We do NOT assign back to X_woe[col] yet to avoid dtype conflict
                mapped_series = binned_series.map(self.woe_maps[col])
                
                # 3. Convert to Numeric and Fill NaNs
                # This ensures the final series is Float, not Categorical
                X_woe[col] = pd.to_numeric(mapped_series, errors='coerce').fillna(0)
            else:
                X_woe[col] = 0.0
        return X_woe

# Execution Logic
def run_pipeline(data_path):
    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # 2. Pre-processing
    processing_pipeline = Pipeline([
        ('time_extraction', TimeFeatureExtractor()),
        ('categorical_encoding', CategoricalEncoder(columns=['ChannelId', 'ProductCategory', 'PricingStrategy'])),
        ('aggregation', CustomerAggregator()),
        ('missing_imputation', MissingValueHandler(strategy='mean')),
    ])
    
    print("Running Pre-processing and Aggregation...")
    X_processed = processing_pipeline.fit_transform(df)
    
    # 3. Target Creation
    score = (X_processed['Total_Transaction_Amount'] / X_processed['Total_Transaction_Amount'].max()) + \
            (X_processed['Transaction_Count'] / X_processed['Transaction_Count'].max())
    
    X_processed['RiskLabel'] = (score >= score.quantile(0.75)).astype(int)
    
    X = X_processed.drop(columns=['AccountId', 'RiskLabel'])
    y = X_processed['RiskLabel']
    
    # 4. WoE and Scaling
    ml_pipeline = Pipeline([
        ('woe', CustomWoETransformer(target_col='RiskLabel')),
        ('scaler', StandardScaler())
    ])
    
    print("Running WoE and Standardization...")
    X_final = ml_pipeline.fit_transform(X, y)
    
    final_df = pd.DataFrame(X_final, columns=X.columns)
    final_df['AccountId'] = X_processed['AccountId'].values
    final_df['RiskLabel'] = y.values
    
    print(f"Transformation Complete. Shape: {final_df.shape}")
    return final_df

if __name__ == "__main__":
    data_path = "../data/raw/data.csv" 
    output_path = "../data/processed/model_ready_data.csv"
    try:

        processed_data = run_pipeline(data_path)
        # print(processed_data.head())
        processed_data.to_csv(output_path, index=False)
        print("Data saved successfully.")
    except FileNotFoundError:
        print("Data file not found. Please check the path.")