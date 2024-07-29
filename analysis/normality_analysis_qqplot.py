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

# Function to perform normality tests and plots
def check_normality(ax1, ax2, data, feature, technique):
    print(f"Checking normality for {feature} in {technique}")
    
    # Shapiro-Wilk Test
    stat, p_value = stats.shapiro(data[feature])
    print(f"Shapiro-Wilk Test: Statistics={stat}, p-value={p_value}")
    
    sns.histplot(data[feature], color="skyblue", kde=True, ax = ax1)
    ax1.set_title(f'Histogram and KDE of {feature} in {technique}')
    ax1.set_xlabel(feature, fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    stats.probplot(data[feature], dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot of {feature} in {technique}')
    ax2.set_xlabel('Theoretical Quantiles', fontsize=10)
    ax2.set_ylabel('Sample Quantiles', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)

# Select the feature and technqiue to check normality
features_to_check = ['Energy(J)', 'AUC']
techniques_to_check = ['Baseline', 'SUS', 'RFE-10']

# Perform normality checks for each technique and the baseline
fig_energy, axes_energy = plt.subplots(2, 3, figsize=(12.5, 7))
fig_AUC, axes_AUC = plt.subplots(2, 3, figsize=(12.5, 7))
for i, technique in enumerate(techniques_to_check):
    technique_data = all_data[all_data['Technique'] == technique]

    check_normality(axes_energy[0, i], axes_energy[1, i], technique_data, 'Energy(J)', technique)
    check_normality(axes_AUC[0, i], axes_AUC[1, i], technique_data, 'AUC', technique)

fig_energy.tight_layout()
fig_AUC.tight_layout()
# fig_energy.savefig(f'normality_check_energy.pdf')
# fig_AUC.savefig(f'normality_check_AUC.pdf')
plt.show()
