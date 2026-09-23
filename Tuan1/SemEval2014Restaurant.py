# -*- coding: utf-8 -*-
# =============================================================================
# Sentiment Analysis trên SemEval 2014 Restaurant Dataset
# Mô hình: Naive Bayes, Logistic Regression (+ baseline: Linear SVM, Random Forest)
# Evaluation: F1, Accuracy, ROC-AUC, Precision, Recall
# =============================================================================

import os
import sys
import warnings

# Fix encoding cho Windows console
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report
)

# --- Các mô hình ML ---
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# =============================================================================
# 1. ĐỌC DỮ LIỆU
# =============================================================================
print("=" * 70)
print("1. ĐỌC DỮ LIỆU")
print("=" * 70)

# Đường dẫn dataset (đã download từ Kaggle)
DATA_DIR = os.path.join(
    "datasets", "charitarth",
    "semeval-2014-task-4-aspectbasedsentimentanalysis",
    "versions", "4"
)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Đọc file CSV training
train_file = os.path.join(DATA_DIR, "Restaurants_Train_v2.csv")
df = pd.read_csv(train_file)

print(f"Số mẫu ban đầu: {len(df)}")
print(f"Các cột: {df.columns.tolist()}")
print(f"\nPhân bố nhãn:")
print(df['polarity'].value_counts())

# =============================================================================
# 2. TIỀN XỬ LÝ DỮ LIỆU
# =============================================================================
print("\n" + "=" * 70)
print("2. TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 70)

# Loại bỏ các hàng bị thiếu dữ liệu
df = df.dropna(subset=['Sentence', 'polarity'])

# Gộp Sentence + Aspect Term để tạo feature text (giữ ngữ cảnh aspect)
df['text'] = df['Sentence'].astype(str) + " " + df['Aspect Term'].astype(str)

# Chuẩn hóa text cơ bản (lowercase)
df['text'] = df['text'].str.lower().str.strip()

print(f"Số mẫu sau tiền xử lý: {len(df)}")

# Encode nhãn polarity
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['polarity'])
class_names = label_encoder.classes_
n_classes = len(class_names)

print(f"Các nhãn: {class_names.tolist()}")
print(f"Số lớp: {n_classes}")

# =============================================================================
# 3. CHIA TẬP TRAIN / TEST
# =============================================================================
print("\n" + "=" * 70)
print("3. CHIA TẬP TRAIN / TEST (80% - 20%)")
print("=" * 70)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} mẫu")
print(f"Test:  {len(X_test)} mẫu")

# Phân bố nhãn Train/Test
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phân bố dữ liệu SemEval 2014 Restaurant", fontsize=16, fontweight='bold')

colors = ['#4A90D9', '#50B86C', '#F5A623', '#E74C6F']

for ax, data, title in [
    (axes[0], y_train, "Phân bố nhãn - Tập Train"),
    (axes[1], y_test,  "Phân bố nhãn - Tập Test")
]:
    counts = pd.Series(data).map(lambda x: class_names[x]).value_counts()
    counts = counts.reindex(class_names)
    bars = ax.bar(counts.index, counts.values, color=colors[:n_classes], edgecolor='white')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel("Số lượng")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "label_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Đã lưu biểu đồ phân bố nhãn → results/label_distribution.png")

# =============================================================================
# 4. TRÍCH XUẤT ĐẶC TRƯNG (TF-IDF)
# =============================================================================
print("\n" + "=" * 70)
print("4. TRÍCH XUẤT ĐẶC TRƯNG (TF-IDF)")
print("=" * 70)

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),      # Unigram + Bigram
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Số features TF-IDF: {X_train_tfidf.shape[1]}")
print(f"Shape Train: {X_train_tfidf.shape}")
print(f"Shape Test:  {X_test_tfidf.shape}")

# =============================================================================
# 5. XÂY DỰNG VÀ HUẤN LUYỆN CÁC MÔ HÌNH
# =============================================================================
print("\n" + "=" * 70)
print("5. XÂY DỰNG VÀ HUẤN LUYỆN CÁC MÔ HÌNH")
print("=" * 70)

# Định nghĩa các mô hình
models = {
    # --- Mô hình chính (yêu cầu đề bài) ---
    "Multinomial Naïve Bayes": MultinomialNB(alpha=1.0),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs', random_state=42
    ),
    # --- Mô hình baseline (để so sánh) ---
    "Linear SVM": LinearSVC(max_iter=2000, C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=50, random_state=42, n_jobs=-1
    ),
}

# Dictionary lưu kết quả
results = {}

for name, model in models.items():
    print(f"\n--- Training: {name} ---")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    # Tính các metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Tính ROC-AUC (cần probability scores)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Lấy probability/decision scores cho ROC
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test_tfidf)
    elif hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_test_tfidf)
    else:
        y_score = label_binarize(y_pred, classes=range(n_classes))

    try:
        roc_macro = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
    except Exception:
        roc_macro = float('nan')

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_score': y_score,
        'Accuracy': round(acc, 4),
        'Precision (Macro)': round(prec_macro, 4),
        'Recall (Macro)': round(rec_macro, 4),
        'F1 (Macro)': round(f1_macro, 4),
        'Precision (Weighted)': round(prec_w, 4),
        'Recall (Weighted)': round(rec_w, 4),
        'F1 (Weighted)': round(f1_w, 4),
        'ROC-AUC (Macro)': round(roc_macro, 4),
    }

    # In classification report
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Precision (Macro): {prec_macro:.4f}")
    print(f"  Recall (Macro):    {rec_macro:.4f}")
    print(f"  F1 (Macro):        {f1_macro:.4f}")
    print(f"  ROC-AUC (Macro):   {roc_macro:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# =============================================================================
# 6. ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH
# =============================================================================
print("\n" + "=" * 70)
print("6. ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH")
print("=" * 70)

# Tạo bảng kết quả
metrics_cols = [
    'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)',
    'Precision (Weighted)', 'Recall (Weighted)', 'F1 (Weighted)', 'ROC-AUC (Macro)'
]
df_results = pd.DataFrame(
    {name: {m: results[name][m] for m in metrics_cols} for name in results}
).T
df_results.index.name = 'Model'

print("\n📊 BẢNG KẾT QUẢ TỔNG HỢP:")
print(df_results.to_string())

# Lưu CSV
df_results.to_csv(os.path.join(RESULTS_DIR, "model_results.csv"))
print(f"\n✅ Đã lưu kết quả → results/model_results.csv")

# --- 6.1 Biểu đồ so sánh các metrics chính ---
compare_metrics = ['Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)', 'ROC-AUC (Macro)']
model_names = list(results.keys())
bar_colors = ['#4A90D9', '#50B86C', '#F5A623', '#E74C6F']

fig, ax = plt.subplots(figsize=(16, 7))
fig.suptitle("So sánh các mô hình ML - Sentiment Analysis\n(SemEval 2014 Restaurant Dataset)",
             fontsize=16, fontweight='bold')

x = np.arange(len(compare_metrics))
width = 0.18
offsets = np.arange(len(model_names)) - (len(model_names) - 1) / 2

for i, name in enumerate(model_names):
    vals = [df_results.loc[name, m] for m in compare_metrics]
    bars = ax.bar(x + offsets[i] * width, vals, width,
                  label=name, color=bar_colors[i], edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(compare_metrics, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Đã lưu biểu đồ so sánh → results/model_comparison.png")

# --- 6.2 Confusion Matrix ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Confusion Matrix - Các mô hình ML\n(SemEval 2014 Restaurant Dataset)",
             fontsize=16, fontweight='bold')

for ax, name in zip(axes.flatten(), model_names):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel("Dự đoán (Predicted)")
    ax.set_ylabel("Thực tế (Actual)")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Đã lưu confusion matrices → results/confusion_matrices.png")

# --- 6.3 ROC Curves (One-vs-Rest) ---
y_test_bin = label_binarize(y_test, classes=range(n_classes))

fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
fig.suptitle("ROC Curves (One-vs-Rest) - Các mô hình ML\n(SemEval 2014 Restaurant Dataset)",
             fontsize=16, fontweight='bold')

for class_idx in range(n_classes):
    ax = axes[class_idx]
    ax.set_title(f"ROC Curve: {class_names[class_idx]}", fontsize=12, fontweight='bold')

    for i, name in enumerate(model_names):
        y_score = results[name]['y_score']

        # Xử lý trường hợp score không cùng shape
        if y_score.ndim == 1:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score[:, class_idx])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=bar_colors[i], linewidth=2,
                    label=f"{name} (AUC={roc_auc_val:.3f})")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.500)')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Đã lưu ROC curves → results/roc_curves.png")

# =============================================================================
# 7. KẾT LUẬN
# =============================================================================
print("\n" + "=" * 70)
print("7. KẾT LUẬN")
print("=" * 70)

# Tìm model tốt nhất theo từng metric
print("\n🏆 Model tốt nhất theo từng metric:")
for metric in compare_metrics:
    best_model = df_results[metric].idxmax()
    best_val = df_results[metric].max()
    print(f"  {metric:25s}: {best_model} ({best_val:.4f})")

# So sánh Naive Bayes vs Logistic Regression (2 model chính)
print("\n📊 SO SÁNH 2 MÔ HÌNH CHÍNH:")
print("-" * 50)
nb_results = df_results.loc["Multinomial Naïve Bayes"]
lr_results = df_results.loc["Logistic Regression"]

comparison_df = pd.DataFrame({
    'Naïve Bayes': nb_results[compare_metrics],
    'Logistic Regression': lr_results[compare_metrics],
    'Chênh lệch': lr_results[compare_metrics] - nb_results[compare_metrics]
})
print(comparison_df.to_string())

print("\n" + "=" * 70)
print("✅ HOÀN TẤT! Tất cả kết quả đã được lưu trong thư mục 'results/'")
print("=" * 70)