# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_csv("/content/dia_imb2.csv")
df.fillna(df.mode().iloc[0], inplace=True)

# Encode categorical variables
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X, y = df.drop(columns=['readmitted']), df['readmitted']

# Handle class imbalance using SMOTE
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)

# Scale features
X_scaled = StandardScaler().fit_transform(X_resampled)

# Reshape for BiLSTM
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Define BiLSTM Model Function
def create_bilstm_model():
    model = Sequential([
        Input(shape=(1, X_scaled.shape[2])),
        Bidirectional(LSTM(64, activation='relu', return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32, activation='relu')),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 10-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Store results
accuracy_list, precision_list, recall_list, f1_list, auroc_list = [], [], [], [], []

# Store confusion matrix values
final_y_true, final_y_pred, final_y_pred_prob = [], [], []

for train_idx, val_idx in kfold.split(X_scaled, y_resampled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y_resampled.iloc[train_idx], y_resampled.iloc[val_idx]

    # Train BiLSTM
    bilstm_model = create_bilstm_model()
    history = bilstm_model.fit(X_train, y_train, epochs=70, batch_size=64, verbose=0, validation_split=0.1)

    # Extract BiLSTM features
    bilstm_features_train = bilstm_model.predict(X_train)
    bilstm_features_val = bilstm_model.predict(X_val)

    # Train XGBoost on BiLSTM features
    xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, random_state=42, tree_method='hist', device='cuda')
    xgb_model.fit(bilstm_features_train, y_train)

    # Evaluate Hybrid Model
    y_pred = xgb_model.predict(bilstm_features_val)
    y_pred_prob = xgb_model.predict_proba(bilstm_features_val)[:, 1]

    accuracy_list.append(accuracy_score(y_val, y_pred))
    precision_list.append(precision_score(y_val, y_pred))
    recall_list.append(recall_score(y_val, y_pred))
    f1_list.append(f1_score(y_val, y_pred))
    auroc_list.append(roc_auc_score(y_val, y_pred_prob))

    # Store for final confusion matrix
    final_y_true.extend(y_val)
    final_y_pred.extend(y_pred)
    final_y_pred_prob.extend(y_pred_prob)

# Print 10-Fold Cross-Validation Results
print(f"BiLSTM-XGBoost Hybrid Model - 10-Fold CV Results:")
print(f"Mean Accuracy: {np.mean(accuracy_list):.4f} (±{np.std(accuracy_list):.4f})")
print(f"Mean Precision: {np.mean(precision_list):.4f} (±{np.std(precision_list):.4f})")
print(f"Mean Recall: {np.mean(recall_list):.4f} (±{np.std(recall_list):.4f})")
print(f"Mean F1 Score: {np.mean(f1_list):.4f} (±{np.std(f1_list):.4f})")
print(f"Mean AUROC: {np.mean(auroc_list):.4f} (±{np.std(auroc_list):.4f})")