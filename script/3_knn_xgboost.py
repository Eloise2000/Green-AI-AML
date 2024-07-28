import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image
from scipy.stats import skew
from matplotlib.transforms import Bbox

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import NearMiss

import time
import warnings
import os

warnings.filterwarnings("ignore")

# Set a random seed based on current time
random_seed = int(time.time()) % 2**32
print(f"Random seed used: {random_seed}")

''' Read pre-processed data '''
df = pd.read_csv("./data/SAML-D_processed.csv")

''' Split the data'''
X = df.drop(columns=['Is_laundering'])
y = df['Is_laundering']

# Step 1: Split the dataset into 80% training and 20% temporary (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_seed)

# Step 2: Split the temporary set into 50% validation and 50% test (each 10% of the original data)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_seed)

# # Check the distribution of the classes in the splits
# print("Class distribution in training set before undersampling:", pd.Series(y_train).value_counts())
# print("Class distribution in validation set:", pd.Series(y_validation).value_counts())
# print("Class distribution in test set:", pd.Series(y_test).value_counts())

# Step 3: Apply RandomUnderSampler to the training set
undersample = NearMiss(version=3)
X_train, y_train = undersample.fit_resample(X_train, y_train)

# # Check the distribution of the classes after undersampling
# print("Class distribution in training set after undersampling:", pd.Series(y_train).value_counts())

# # The validation and test sets remain unchanged
# print("Class distribution in validation set remains:", pd.Series(y_validation).value_counts())
# print("Class distribution in test set remains:", pd.Series(y_test).value_counts())

''' Hyperparameter Search Training Model '''
# param_grid = {
#     'max_depth': [6, 8, 9, 12],
#     'eta': [0.05, 0.07, 1],
#     'n_estimators': [150, 180, 200],
# }

# # xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_seed)
# xgb = XGBClassifier(use_label_encoder=False, eval_metric='aucpr', random_state=random_seed)

# grid_search = GridSearchCV(
#     estimator=xgb,
#     param_grid=param_grid,
#     scoring='roc_auc',
#     # scoring='precision',
#     cv=4,
#     verbose=2
# )

# start_time = time.time()
# grid_search.fit(X_train, y_train)
# end_time = time.time()

# print(f'Test spend on training: {(end_time-start_time)*1000} ms')
# print("Best Parameters: ", grid_search.best_params_)

# best_model = grid_search.best_estimator_

''' Pure Training Model '''
# Initialize model with fixed hyperparameters
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='aucpr',  # Evaluation metric suitable for imbalanced data
    max_depth=9,
    eta=0.1,
    n_estimators=200,
    random_state=random_seed
)

# Time the fitting process
start_time = time.time()
xgb.fit(X_train, y_train)
end_time = time.time()

best_model = xgb

''' Save the trained model '''
# filename = os.path.basename(__file__)
# save_path = '/home/eloise/eloise/model/'+filename[:-3]+'.model'
# best_model.save_model(save_path)
# print(f"Best model saved in {save_path}")

''' Test the model '''
val_predictions = best_model.predict_proba(X_validation)[:, 1]
val_auc = roc_auc_score(y_validation, val_predictions)

test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_probabilities)

fpr, tpr, thresholds = roc_curve(y_test, test_probabilities)

# Confusion Matrix, TPR, and FPR at around a TPR of 0.9
desired_tpr = 0.8
closest_threshold = thresholds[np.argmin(np.abs(tpr - desired_tpr))]

y_pred = (test_probabilities >= closest_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()
fpr_cm = fp / (fp + tn)
tpr_cm = tp / (tp + fn)

# Compute additional metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print out the metrics
print(f"----- Model Performance Metrics (Desired TPR is around {desired_tpr*100}%) -----")
print(f'Time spent on training: {(end_time - start_time) * 1000:.2f} ms')
print(f"Validation AUC: {val_auc:.3f}")
print(f"Test AUC: {test_auc:.3f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"False Positive Rate (FPR): {fpr_cm:.3f}")
print(f"True Positive Rate (TPR): {tpr_cm:.3f}")