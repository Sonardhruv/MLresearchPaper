from sklearn.metrics import matthews_corrcoef

# Compute confusion matrix
cm = confusion_matrix(final_y_true, final_y_pred)

# Extract TN, FP, FN, TP
TN, FP, FN, TP = cm.ravel()

# Compute additional metrics
error_rate = 1 - accuracy_score(final_y_true, final_y_pred)
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # Avoid division by zero
mcc = matthews_corrcoef(final_y_true, final_y_pred)

# Print updated results
print(f"BiLSTM-XGBoost Hybrid Model - 10-Fold CV Results:")
print(f"Mean Accuracy: {np.mean(accuracy_list):.4f} (±{np.std(accuracy_list):.4f})")
print(f"Mean Precision: {np.mean(precision_list):.4f} (±{np.std(precision_list):.4f})")
print(f"Mean Recall: {np.mean(recall_list):.4f} (±{np.std(recall_list):.4f})")
print(f"Mean F1 Score: {np.mean(f1_list):.4f} (±{np.std(f1_list):.4f})")
print(f"Mean AUROC: {np.mean(auroc_list):.4f} (±{np.std(auroc_list):.4f})")
print(f"Error Rate: {error_rate:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")