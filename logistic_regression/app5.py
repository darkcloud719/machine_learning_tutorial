import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 1. True labels (ground truth)
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])

# 2. Predicted labels (model output)
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

# 3. Predicted Probabilities (for AUC)
y_prob = np.array([0.9, 0.2, 0.8, 0.4, 0.1, 0.7, 0.6, 0.3])

# 1. Accuracy (準確率)
accuracy = accuracy_score(y_true, y_pred)

# 2 Error Rate (錯誤率)
error_rate = 1 - accuracy

# 3. Precision (精確率)
precision = precision_score(y_true, y_pred)

# 4. Recall (召回率)
recall = recall_score(y_true, y_pred)

# 5. F1-score (F1分數)
f1 = f1_score(y_true, y_pred)

# 6. AUC (ROC-AUC)
auc = roc_auc_score(y_true, y_prob)

# 7. Confusion Matrix (幫助理解)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# print results

print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)

print("\nConfusion Matrix:")
print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)