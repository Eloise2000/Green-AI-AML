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

# Select the feature and technqiue to check normality
features_to_check = ['Energy(J)', 'AUC']
techniques_to_check = ['Baseline', 'SUS', 'RFE-10']

''' Hypothesis Testing '''
# Function to compare techniques against the baseline
def compare_with_baseline(baseline_data, technique_data, feature):
    print(f"\nComparing {feature} between Baseline and {technique_data['Technique'].iloc[0]}")
    
    # Check if both datasets are normally distributed
    baseline_stat, baseline_p = stats.shapiro(baseline_data[feature])
    technique_stat, technique_p = stats.shapiro(technique_data[feature])
    
    if baseline_p > 0.05 and technique_p > 0.05:
        # Perform paired t-test
        stat, p_value = stats.ttest_rel(baseline_data[feature], technique_data[feature])
        print(f"Paired t-test: Statistics={stat}, p-value={p_value}")
    else:
        # Perform Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(baseline_data[feature], technique_data[feature])
        print(f"Wilcoxon signed-rank test: Statistics={stat}, p-value={p_value}")

# Separate baseline data
baseline_data = all_data[all_data['Technique'] == 'Baseline']

# Compare each technique with the baseline
for technique in techniques_to_check:
    if technique != "Baseline":
        technique_data = all_data[all_data['Technique'] == technique]
        for feature in features_to_check:
            compare_with_baseline(baseline_data, technique_data, feature)
