
#Task 2 - Exploratory Data Analysis (EDA)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your dataset
df = pd.read_csv('../Data/data.csv')

#Task 3 - Default estimator and WoE binning
#Construct a default estimator (proxy): For simplicity, we will create RFMS scores.

def calculate_rfms(df):
    rfm = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'count'],  # Total Revenue, Count of Orders
        'TransactionStartTime': lambda x: (pd.Timestamp.now().tz_localize(None) - x.max()).days  # Recency as days since last transaction
    })

    rfm.columns = ['Total_Spent', 'Total_Transactions', 'Recency']
    # Score assignment could be expanded based on business logic
    rfm['RFMS_Score'] = (rfm['Recency'] + rfm['Total_Spent'] + rfm['Total_Transactions']).mean()
    return rfm