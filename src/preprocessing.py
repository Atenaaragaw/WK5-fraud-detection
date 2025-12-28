import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# SETUP PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, '../data/raw/')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, '../data/processed/')
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def load_data():
    fraud = pd.read_csv(os.path.join(RAW_DATA_PATH, 'Fraud_Data.csv'))
    ip_to_country = pd.read_csv(os.path.join(RAW_DATA_PATH, 'IpAddress_to_Country.csv'))
    credit = pd.read_csv(os.path.join(RAW_DATA_PATH, 'creditcard.csv'))
    return fraud, ip_to_country, credit

def map_ip_to_country(fraud_df, ip_df):
    # Ensure float64 for large IP integers
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(float)
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(float)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(float)
    
    fraud_df = fraud_df.sort_values('ip_address')
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    
    merged_df = pd.merge_asof(
        fraud_df, 
        ip_df, 
        left_on='ip_address', 
        right_on='lower_bound_ip_address'
    )
    
    merged_df['country'] = np.where(
        merged_df['ip_address'] <= merged_df['upper_bound_ip_address'],
        merged_df['country'],
        'Unknown'
    )
    return merged_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])

def engineer_features(df):
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Behavioral Velocity Features
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Transaction Frequency (Bot/Sybil Detection)
    df['user_freq'] = df.groupby('user_id')['user_id'].transform('count')
    df['device_freq'] = df.groupby('device_id')['user_id'].transform('count')
    
    return df

def transform_data(df):
    le = LabelEncoder()
    cat_cols = ['source', 'browser', 'sex', 'country']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    scaler = StandardScaler()
    num_cols = ['purchase_value', 'age', 'time_since_signup', 'device_freq', 'user_freq']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df

def main():
    fraud, ip_to_country, credit = load_data()
    fraud = fraud.drop_duplicates().dropna()
    
    fraud_merged = map_ip_to_country(fraud, ip_to_country)
    fraud_featured = engineer_features(fraud_merged)
    fraud_final = transform_data(fraud_featured)
    
    fraud_final.to_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_fraud_data.csv'), index=False)
    credit.to_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_credit_data.csv'), index=False)
    print("Task 1 Preprocessing Complete.")

if __name__ == "__main__":
    main()