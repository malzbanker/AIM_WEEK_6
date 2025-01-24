#Task 2 - Exploratory Data Analysis (EDA)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your dataset
df = pd.read_csv('../Data/data.csv')

#Task 3 - Feature Engineering

# 1. Create Aggregate Features
df['Total_Transaction_Amount'] = df.groupby('CustomerId')['Amount'].transform('sum')
df['Average_Transaction_Amount'] = df.groupby('CustomerId')['Amount'].transform('mean')
df['Transaction_Count'] = df.groupby('CustomerId')['TransactionId'].transform('count')
df['Std_Dev_Transaction_Amount'] = df.groupby('CustomerId')['Amount'].transform('std')

# 2. Extract Features from TransactionStartTime
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
df['Transaction_Day'] = df['TransactionStartTime'].dt.day
df['Transaction_Month'] = df['TransactionStartTime'].dt.month
df['Transaction_Year'] = df['TransactionStartTime'].dt.year

# 3. Encode Categorical Variables
# Check if these columns exist in your DataFrame
categorical_features_list = ['ProductCategory', 'CurrencyCode', 'FraudResult', 'TransactionId'] 
# Only include 'MerchantName' if it's in the DataFrame's columns and hasn't been one-hot encoded yet
if 'MerchantName' in df.columns and 'MerchantName_1' not in df.columns:  # Check for one-hot encoded version
    categorical_features_list.append('MerchantName')

# Create a copy of the DataFrame for categorical features BEFORE one-hot encoding
categorical_features = df[categorical_features_list].copy() if all(col in df.columns for col in categorical_features_list) else pd.DataFrame()

# Now apply one-hot encoding to the original DataFrame, but only if the columns haven't been encoded already
columns_to_encode = [col for col in categorical_features.drop('TransactionId', axis=1, errors='ignore').columns if col in df.columns and f'{col}_1' not in df.columns]
df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

# 4. Handle Missing Values
# Impute numerical columns with mean, EXCLUDING non-numeric columns
numeric_columns = df.select_dtypes(include=np.number).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())  
# Alternatively, drop if few
df.dropna(inplace=True)

# 5. Normalize/Standardize Numerical Features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Redefine numerical_features to include only the current numerical columns
numerical_features = df.select_dtypes(include=np.number).columns  
# Exclude the one-hot encoded and other engineered features
numerical_features = [f for f in numerical_features if f in ['CountryCode', 'Amount', 'Value']]  
# Now apply scaling 
df[numerical_features] = scaler.fit_transform(df[numerical_features])