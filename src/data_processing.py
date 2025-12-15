import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Custom Transformations
class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts Hour, Day, Month, and Year from TransactionStartTime.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if 'TransactionStartTime' not in X.columns:
            raise KeyError("Column 'TransactionStartTime' not found in dataset. Check inputs.")
        
        
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
        # Define Snapshot Date for Recency (Max Date + 1 Day)
        if 'TransactionStartTime' in X.columns:
            # Fall back if datetime conversion hasn't happended yet
            snapshot_date = pd.to_datetime(X['TransactionStartTime']).max() + pd.Timedelta(days=1)

        else:
            # Should not happen if TimeFeatureExtractor ran, but safety first
            raise ValueError("TransactionStartTime missing for Recency calculation")

        
        agg_rules = {
            "TransactionStartTime": 'max', # Recency
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

        df_agg['Recency'] = (snapshot_date - df_agg['TransactionStartTime_max']).dt.days
        
        df_agg.rename(columns={
            'Value_sum': 'Monetary_Total',
            'Value_mean': 'Monetary_Mean',
            'Value_count': 'Frequency',
            'Value_std': 'Monetary_Std'
        }, inplace=True)

        # Drop the helper column used for Recency
        df_agg.drop(columns=['TransactionStartTime_max'], inplace=True, errors='ignore')
        
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

class RiskProxyLabeler(BaseEstimator, TransformerMixin):
    """
    Creates: 'RiskLabel' target using K-Means CLustering on RFM features.
    """
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = None
        self.high_risk_cluster_label = None

    def fit(self, X, y=None):
        # We need RFM columns to cluster
        rfm_cols = ['Recency', 'Frequency', 'Monetary_Total']

        missing = [c for c in rfm_cols if c not in X.columns]
        if missing:
            raise KeyError(f"Missing columns for Clustering: {missing}")

        # Log Transform and Scale (Crucial for K-Means with financial data)
        # Add 1 to avoid log(0)
        X_rfm = np.log1p(X[rfm_cols])
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X_rfm)

        # Perform CLustering
        self.kmeans =KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        

        # Analyze clusters to identify "High Risk"
        # Combine labels with original RFM vlaues to check means
        analysis_df = pd.DataFrame(X[rfm_cols].copy())
        analysis_df['Cluster'] = clusters

        cluster_stats = analysis_df.groupby('Cluster').mean()

        # Definition of High Risk (Disengaged):
        # High Recency (Inactive for a long time)
        # Low Frequency (Rarely buys)
        # Low Monetary (Spends little)
        
        # We create a simple "Risk Score" for ranking: Recency - Frequency - Monetary
        # (Since we want Max Recency and Min Freq/Monetary)
        # Note: We use the scaled means for fair comparison if magnitudes differ wildly, 
        # but here raw means usually suffice to spot the "inactive" group.
        
        # Let's verify using the stats:
        print("\n--- Cluster Analysis for Risk Proxy ---")
        print(cluster_stats)
        
        # Logic: The cluster with the HIGHEST Recency is usually the "Churned/Disengaged" one.
        # Alternatively, we can look for the lowest Frequency.
        self.high_risk_cluster_label = cluster_stats['Recency'].idxmax()
        
        print(f"\nIdentified Cluster {self.high_risk_cluster_label} as High Risk (Disengaged).")
        return self

    def transform(self, X):
        rfm_cols = ['Recency', 'Frequency', 'Monetary_Total']
        X_rfm = np.log1p(X[rfm_cols])

        X_scaled = self.scaler.transform(X_rfm)
        
        clusters = self.kmeans.predict(X_scaled)
        
        # Assign 1 if cluster is the high_risk one, else 0
        X = X.copy()
        X['RiskLabel'] = (clusters == self.high_risk_cluster_label).astype(int)
        
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
        
        total_bad = df['target'].sum()
        total_good = df['target'].count() - total_bad
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

                grouped['bad'] = grouped['sum']
                grouped['good'] = grouped['count'] - grouped['sum']
                
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
    
    # 3. Proxy target Engineering
    print("Running Proxy Target Engineering (K-Means)...")
    labeler = RiskProxyLabeler(n_clusters=3, random_state=42)
    # Fit finds the clusters and identifies the risky one
    labeler.fit(X_processed)
    # Transform assigns the "Risklabel" column
    X_labeled = labeler.transform(X_processed)

    print("Class Balance (is_high_risk):")
    print(X_labeled['RiskLabel'].value_counts(normalize=True))

    
    X = X_labeled.drop(columns=['AccountId', 'RiskLabel'])
    y = X_labeled['RiskLabel']
    
    # 4. WoE and Scaling
    ml_pipeline = Pipeline([
        ('woe', CustomWoETransformer(target_col='RiskLabel')),
        ('scaler', StandardScaler())
    ])
    
    print("Running WoE and Standardization...")
    X_final = ml_pipeline.fit_transform(X, y)
    
    final_df = pd.DataFrame(X_final, columns=X.columns)
    final_df['AccountId'] = X_labeled['AccountId'].values
    final_df['RiskLabel'] = y.values
    
    print(f"Transformation Complete. Shape: {final_df.shape}")
    return final_df

if __name__ == "__main__":
    data_path = "./data/raw/data.csv" 
    output_path = "./data/processed/model_ready_data.csv"
    try:

        processed_data = run_pipeline(data_path)
        # print(processed_data.head())
        processed_data.to_csv(output_path, index=False)
        print("Data saved successfully.")
    except FileNotFoundError:
        print("Data file not found. Please check the path.")