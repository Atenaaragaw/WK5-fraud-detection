import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. SETUP PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, '../data/raw/')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, '../data/processed/')

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def load_data():
    print("--- Loading datasets ---")
    fraud = pd.read_csv(os.path.join(RAW_DATA_PATH, 'Fraud_Data.csv'))
    ip_to_country = pd.read_csv(os.path.join(RAW_DATA_PATH, 'IpAddress_to_Country.csv'))
    credit = pd.read_csv(os.path.join(RAW_DATA_PATH, 'creditcard.csv'))
    return fraud, ip_to_country, credit

def map_ip_to_country(fraud_df, ip_df):
    print("--- Geolocation Integration: Mapping IP to Country ---")
    # Requirements: Convert IP to numeric and sort for range lookup
    fraud_df = fraud_df.sort_values('ip_address')
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    
    # Efficient range join using merge_asof
    merged_df = pd.merge_asof(
        fraud_df, 
        ip_df, 
        left_on='ip_address', 
        right_on='lower_bound_ip_address'
    )
    
    # Boundary check: Ensure IP is within the upper bound
    merged_df['country'] = np.where(
        merged_df['ip_address'] <= merged_df['upper_bound_ip_address'],
        merged_df['country'],
        'Unknown'
    )
    
    return merged_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])

def engineer_features(df):
    print("--- Feature Engineering: Velocity and Time ---")
    # Correct Data Types
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # 1. Time-based features
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # 2. Transaction frequency/velocity
    # Count occurrences of same device or IP to spot bot patterns
    df['device_freq'] = df.groupby('device_id')['user_id'].transform('count')
    df['ip_freq'] = df.groupby('ip_address')['user_id'].transform('count')
    
    return df

def transform_data(df):
    print("--- Data Transformation: Encoding and Scaling ---")
    
    # 1. Handle Missing Values (Requirement: Impute or Drop)
    # Since 'country' might have 'Unknown' from the merge, we keep it as a category
    df = df.dropna(subset=['user_id']) # Drop rows without core ID
    
    # 2. Encode Categorical Features (Requirement: One-Hot or Label Encoding)
    # We'll use Label Encoding for high-cardinality columns like country
    le = LabelEncoder()
    cat_cols = ['source', 'browser', 'sex', 'country']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # 3. Normalize/Scale Numerical Features (Requirement: StandardScaler)
    scaler = StandardScaler()
    num_cols = ['purchase_value', 'age', 'time_since_signup', 'device_freq', 'ip_freq']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df

def main():
    # Load
    fraud, ip_to_country, credit = load_data()
    
    # Clean: Remove duplicates
    fraud = fraud.drop_duplicates()
    
    # Task 1.3: Geolocation
    fraud_merged = map_ip_to_country(fraud, ip_to_country)
    
    # Task 1.4: Feature Engineering
    fraud_featured = engineer_features(fraud_merged)
    
    # Task 1.5: Data Transformation (Scaling/Encoding)
    fraud_final = transform_data(fraud_featured)
    
    # Save processed files
    print(f"--- Saving processed data to {PROCESSED_DATA_PATH} ---")
    fraud_final.to_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_fraud_data.csv'), index=False)
    
    # Credit card data already numerical, so we just move it to processed
    credit.to_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_credit_data.csv'), index=False)
    
    print("SUCCESS: Task 1 Preprocessing Complete!")

if __name__ == "__main__":
    main()