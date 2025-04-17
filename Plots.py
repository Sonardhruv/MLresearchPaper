import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# -------- PLOT 1: Feature Correlation Heatmap --------
X_corr = pd.DataFrame(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[2])).corr()

# Now plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(X_corr, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)

# Add title and adjust labels
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# -------- PLOT 2: XGBoost Feature Importance --------
plt.figure(figsize=(10, 6))
plot_importance(xgb_model, importance_type="weight")  # Use trained XGBoost model
plt.title("Feature Importance - XGBoost")
plt.show()

# -------- PLOT 3: BiLSTM Learning Curve --------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("BiLSTM Learning Curve")
plt.legend()
plt.show()

# -------- PLOT 4: Class Distribution Before & After SMOTE --------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x=y, palette="coolwarm")
plt.title("Class Distribution Before SMOTE")

plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled, palette="coolwarm")
plt.title("Class Distribution After SMOTE")
plt.show()

# -------- PLOT 5: Confusion Matrix --------
cm = confusion_matrix(final_y_true, final_y_pred)  # Use stored cross-validation results
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Readmitted', 'Readmitted'], yticklabels=['Not Readmitted', 'Readmitted'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - BiLSTM-XGBoost Model")
plt.show()

# -------- PLOT 6: ROC Curve --------
fpr, tpr, _ = roc_curve(final_y_true, final_y_pred_prob)  # Use probability predictions
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - BiLSTM-XGBoost Model")
plt.legend(loc="lower right")
plt.show()

# -------- PLOT 7: Precision-Recall Curve --------
precision, recall, _ = precision_recall_curve(final_y_true, final_y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='green', lw=2, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - BiLSTM-XGBoost Model")
plt.legend()
plt.show()

# -------- PLOT 8: Cross-Validation Metric Trends (Boxplot) --------
cv_results = {
    "Accuracy": accuracy_list,
    "Precision": precision_list,
    "Recall": recall_list,
    "F1 Score": f1_list,
    "AUROC": auroc_list
}

plt.figure(figsize=(8, 5))
sns.boxplot(data=pd.DataFrame(cv_results))
plt.title("10-Fold Cross-Validation Results - BiLSTM-XGBoost Hybrid Model")
plt.ylabel("Score")
plt.show()