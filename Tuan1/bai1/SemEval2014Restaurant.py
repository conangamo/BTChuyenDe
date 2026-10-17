# -*- coding: utf-8 -*-
# =============================================================================
# Sentiment Analysis trên SemEval 2014 Restaurant Dataset
# Mô hình: Naive Bayes, Logistic Regression (+ baseline: Linear SVM, Random Forest)
# Evaluation: F1, Accuracy, ROC-AUC, Precision, Recall
# Sử dụng dữ liệu đã tiền xử lý + augmentation từ data_preprocessing.py
# =============================================================================

import os
import sys
import warnings

# Fix encoding cho Windows console
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
# 1. ĐỌC DỮ LIỆU ĐÃ TIỀN XỬ LÝ
# =============================================================================
print("=" * 70)
print("1. ĐỌC DỮ LIỆU ĐÃ TIỀN XỬ LÝ (results/processed_data/)")
print("=" * 70)

# Thư mục chứa dữ liệu đã xử lý (output từ data_preprocessing.py)
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))   # .../bai1/
TUAN1_DIR     = os.path.dirname(BASE_DIR)                    # .../Tuan1/
PROCESSED_DIR = os.path.join(TUAN1_DIR, "results", "processed_data")
RESULTS_DIR   = os.path.join(TUAN1_DIR, "results", "bai1")
os.makedirs(RESULTS_DIR, exist_ok=True)


# Đọc 3 tập đã chia sẵn
df_train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
df_val   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
df_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

print(f"Train (goc + augmented) : {len(df_train):>6} mau")
print(f"Validation (goc)        : {len(df_val):>6} mau")
print(f"Test       (goc)        : {len(df_test):>6} mau")

print(f"\nPhan bo nhan — Train:")
print(df_train['polarity'].value_counts().to_string())
print(f"\nPhan bo nhan — Validation:")
print(df_val['polarity'].value_counts().to_string())
print(f"\nPhan bo nhan — Test:")
print(df_test['polarity'].value_counts().to_string())

# =============================================================================
# 2. CHUẨN BỊ FEATURES VÀ NHÃN
# =============================================================================
print("\n" + "=" * 70)
print("2. CHUẨN BỊ FEATURES VÀ NHÃN")
print("=" * 70)

# Dùng cột 'text' đã làm sạch sẵn (Sentence_clean [SEP] Aspect_clean)
X_train_raw = df_train['text'].fillna('')
X_val_raw   = df_val['text'].fillna('')
X_test_raw  = df_test['text'].fillna('')

# Encode nhãn — fit chỉ trên tập train (3 nhãn: negative, neutral, positive)
label_encoder = LabelEncoder()
label_encoder.fit(df_train['polarity'])
class_names = label_encoder.classes_
n_classes   = len(class_names)

y_train = label_encoder.transform(df_train['polarity'])
y_val   = label_encoder.transform(df_val['polarity'])
y_test  = label_encoder.transform(df_test['polarity'])

print(f"Cac nhan : {class_names.tolist()}")
print(f"So lop   : {n_classes}")
print(f"\nX_train  : {len(X_train_raw)} mau")
print(f"X_val    : {len(X_val_raw)} mau")
print(f"X_test   : {len(X_test_raw)} mau")

# Phân bố nhãn 3 tập
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Phan bo du lieu SemEval 2014 Restaurant\n(Train/Val/Test sau tien xu ly & augmentation)",
             fontsize=14, fontweight='bold')

colors_map = {'negative': '#E74C6F', 'neutral': '#4A90D9', 'positive': '#50B86C'}

for ax, (df_part, title) in zip(axes, [
    (df_train[df_train['is_augmented'] == False], "Train (goc, 2503 mau)"),
    (df_val,  "Validation (GIU NGUYEN)"),
    (df_test, "Test (GIU NGUYEN)"),
]):
    counts = df_part['polarity'].value_counts().reindex(class_names, fill_value=0)
    bars = ax.bar(counts.index,
                  counts.values,
                  color=[colors_map.get(l, '#999') for l in counts.index],
                  edgecolor='white', linewidth=1.2)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel("So luong")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(val), ha='center', va='bottom', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "label_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("\n[OK] Da luu bieu do phan bo nhan → results/bai1/label_distribution.png")

# =============================================================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG (TF-IDF)
# =============================================================================
print("\n" + "=" * 70)
print("3. TRICH XUAT DAC TRUNG (TF-IDF)")
print("=" * 70)

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),      # Unigram + Bigram
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# fit CHI tren Train, transform ca 3 tap
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_val_tfidf   = tfidf.transform(X_val_raw)
X_test_tfidf  = tfidf.transform(X_test_raw)

print(f"So features TF-IDF : {X_train_tfidf.shape[1]}")
print(f"Shape Train        : {X_train_tfidf.shape}")
print(f"Shape Validation   : {X_val_tfidf.shape}")
print(f"Shape Test         : {X_test_tfidf.shape}")

# =============================================================================
# 4. XÂY DỰNG VÀ HUẤN LUYỆN CÁC MÔ HÌNH
# =============================================================================
print("\n" + "=" * 70)
print("4. XAY DUNG VA HUAN LUYEN CAC MO HINH")
print("=" * 70)
print("   Train tren tap Train (augmented), danh gia tren tap Validation & Test")

# Định nghĩa các mô hình
models = {
    # --- Mô hình chính (yêu cầu đề bài) ---
    "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs', random_state=42
    ),
    # --- Mô hình baseline (để so sánh) ---
    "Linear SVM": LinearSVC(max_iter=2000, C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=50, random_state=42, n_jobs=-1
    ),
}

# Dictionary lưu kết quả (đánh giá trên Test)
results = {}

for name, model in models.items():
    print(f"\n--- Training: {name} ---")
    model.fit(X_train_tfidf, y_train)

    # ── Đánh giá trên Validation
    y_val_pred = model.predict(X_val_tfidf)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1  = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
    print(f"  [VAL]  Accuracy={val_acc:.4f}  F1(macro)={val_f1:.4f}")

    # ── Đánh giá trên Test
    y_pred = model.predict(X_test_tfidf)

    acc       = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec_w     = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_w      = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_w       = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Tính ROC-AUC
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
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
        'Val Accuracy': round(val_acc, 4),
        'Val F1 (Macro)': round(val_f1, 4),
        'Accuracy': round(acc, 4),
        'Precision (Macro)': round(prec_macro, 4),
        'Recall (Macro)': round(rec_macro, 4),
        'F1 (Macro)': round(f1_macro, 4),
        'Precision (Weighted)': round(prec_w, 4),
        'Recall (Weighted)': round(rec_w, 4),
        'F1 (Weighted)': round(f1_w, 4),
        'ROC-AUC (Macro)': round(roc_macro, 4),
    }

    # In classification report trên Test
    print(f"  [TEST] Accuracy={acc:.4f}  Precision={prec_macro:.4f}  Recall={rec_macro:.4f}  F1={f1_macro:.4f}  ROC-AUC={roc_macro:.4f}")
    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# =============================================================================
# 5. ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH
# =============================================================================
print("\n" + "=" * 70)
print("5. DANH GIA VA SO SANH MO HINH")
print("=" * 70)

# Tạo bảng kết quả
metrics_cols = [
    'Val Accuracy', 'Val F1 (Macro)',
    'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)',
    'Precision (Weighted)', 'Recall (Weighted)', 'F1 (Weighted)', 'ROC-AUC (Macro)'
]
df_results = pd.DataFrame(
    {name: {m: results[name][m] for m in metrics_cols} for name in results}
).T
df_results.index.name = 'Model'

print("\nBANG KET QUA TONG HOP (Test Set):")
print(df_results.to_string())

# Lưu CSV
df_results.to_csv(os.path.join(RESULTS_DIR, "model_results.csv"))
print(f"\n[OK] Da luu ket qua → results/bai1/model_results.csv")

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
print("7. KET LUAN")
print("=" * 70)

# Tìm model tốt nhất theo từng metric (Test)
test_metrics = ['Accuracy', 'F1 (Macro)', 'Precision (Macro)', 'Recall (Macro)', 'ROC-AUC (Macro)']
print("\nModel tot nhat theo tung metric (tren Test):")
for metric in test_metrics:
    best_model = df_results[metric].idxmax()
    best_val   = df_results[metric].max()
    print(f"  {metric:25s}: {best_model} ({best_val:.4f})")

# So sánh Naive Bayes vs Logistic Regression (2 model chính)
print("\nSO SANH 2 MO HINH CHINH (tren Test):")
print("-" * 50)
nb_results = df_results.loc["Multinomial Naive Bayes"]
lr_results = df_results.loc["Logistic Regression"]

comparison_df = pd.DataFrame({
    'Naive Bayes': nb_results[test_metrics],
    'Logistic Regression': lr_results[test_metrics],
    'Chenh lech': lr_results[test_metrics] - nb_results[test_metrics]
})
print(comparison_df.to_string())

print("\n" + "=" * 70)
print("HOAN TAT! Tat ca ket qua da duoc luu trong thu muc 'results/bai1/'")
print("=" * 70)