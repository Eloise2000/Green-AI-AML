import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


metrics = ['AUC', 'Energy(J)', 'Runtime(s)', 'Accuracy', 'False Positive Rate']

# Apply Seaborn theme for a better look
sns.set_theme(style="whitegrid")

# Plot boxplots for each metric
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Technique', y=metric, data=all_data, color="skyblue", showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"10"})
    plt.title(f'Boxplot of {metric} Across Different Techniques Applied', fontsize=16, weight='bold')
    plt.xlabel('Techniques Applied', fontsize=14, weight='bold')
    plt.ylabel(metric, fontsize=14, weight='bold')
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'boxplot_{metric}.pdf')
    # plt.show()