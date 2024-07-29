import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import glob

filefolder = "output_extracted/"
# List of treatments and their corresponding CSV files
techniques = {
    'Baseline': filefolder + 'perf_output_0_normal_xgboost.csv',
    'RUS': filefolder + 'perf_output_1_rus_xgboost.csv',
    'SUS': filefolder + 'perf_output_2_stratified_xgboost.csv',
    'KNN': filefolder + 'perf_output_3_knn_xgboost.csv',
    'RFE-11': filefolder + 'perf_output_4_rfe_xgboost_features_11.csv',
    'RFE-10': filefolder + 'perf_output_4_rfe_xgboost_features_10.csv',
    'RFE-9': filefolder + 'perf_output_4_rfe_xgboost_features_9.csv',
    'RFE-8': filefolder + 'perf_output_4_rfe_xgboost_features_8.csv',
    'RFE-7': filefolder + 'perf_output_4_rfe_xgboost_features_7.csv'
}

# Mapping of original column names to final column names
column_mapping = {
    'Test_AUC': 'AUC',
    'Accuracy': 'Accuracy',
    'Precision': 'Precision',
    'Recall': 'Recall',
    'F1_Score': 'F1',
    'FPR': 'False Positive Rate',
    'TPR': 'TPR',
    'Energy_pkg': 'Energy_pkg',
    'Energy_ram': 'Energy_ram',
    'Time_elapsed': 'Runtime(s)'
}

# Columns of interest (final column names)
columns_of_interest = ['Technique', 'AUC', 'Energy(J)', 'Runtime(s)', 'Recall', 'Accuracy', 'Precision', 'F1', 'False Positive Rate']

# Initialize an empty DataFrame to store all data
all_data = pd.DataFrame()

# Process each treatment
for technique, file in techniques.items():
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Rename the columns according to the mapping
    df = df.rename(columns=column_mapping)
    
    # Compute the Energy feature
    df['Energy(J)'] = df['Energy_pkg'] + df['Energy_ram']
    df['Technique'] = technique

    df = df[columns_of_interest]
    all_data = pd.concat([all_data, df], ignore_index=True)


''' Effect Size Estimation '''
# Separate data for baseline, SUS, and RFE-10
baseline_data = all_data[all_data['Technique'] == 'Baseline'].reset_index(drop=True)
sus_data = all_data[all_data['Technique'] == 'SUS'].reset_index(drop=True)
rfe10_data = all_data[all_data['Technique'] == 'RFE-10'].reset_index(drop=True)

# Define function to calculate Cohen's d
def cohen_d(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

# Merge baseline data with SUS and RFE-10 data on index
merged_sus = pd.merge(baseline_data, sus_data, left_index=True, right_index=True, suffixes=('_baseline', '_sus'))
merged_rfe10 = pd.merge(baseline_data, rfe10_data, left_index=True, right_index=True, suffixes=('_baseline', '_rfe10'))

# Calculate Cohen's d for each comparison
cohen_d_energy_sus = cohen_d(merged_sus['Energy(J)_baseline'], merged_sus['Energy(J)_sus'])
cohen_d_auc_sus = cohen_d(merged_sus['AUC_baseline'], merged_sus['AUC_sus'])
cohen_d_energy_rfe10 = cohen_d(merged_rfe10['Energy(J)_baseline'], merged_rfe10['Energy(J)_rfe10'])
cohen_d_auc_rfe10 = cohen_d(merged_rfe10['AUC_baseline'], merged_rfe10['AUC_rfe10'])

print(f"Cohen's d for Energy(J) between Baseline and SUS: {cohen_d_energy_sus}")
print(f"Cohen's d for AUC between Baseline and SUS: {cohen_d_auc_sus}")
print(f"Cohen's d for Energy(J) between Baseline and RFE-10: {cohen_d_energy_rfe10}")
print(f"Cohen's d for AUC between Baseline and RFE-10: {cohen_d_auc_rfe10}")