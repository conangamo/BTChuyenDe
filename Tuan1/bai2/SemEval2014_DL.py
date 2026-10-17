# -*- coding: utf-8 -*-
# =============================================================================
# BAI 2 — Deep Learning Sentiment Analysis
# SemEval 2014 Restaurant ABSA Dataset
# =============================================================================
# Mo hinh DL:
#   1. MLP  (Multi-Layer Perceptron) tren TF-IDF + Lexicon Features
#   2. BiLSTM (Bidirectional LSTM) voi trainable Word Embeddings
#   3. TextCNN (Kim 2014) — parallel Conv1D voi nhieu kernel sizes
#   4. BiLSTM + Self-Attention
#   5. DL Ensemble (Soft Voting: BiLSTM + TextCNN + BiLSTM-Attention)
# Feature Engineering:
#   - Word Tokenization + Padding
#   - TF-IDF (Unigram + Bigram)
#   - Sentiment Lexicon Features (hand-crafted)
#   - Combined TF-IDF + Lexicon (cho MLP)
# Ensemble Method: Soft Voting tren xac suat softmax cua 3 DL models
# So sanh: ket qua bai1 ML baselines
# =============================================================================

import os
import sys
import warnings

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ─────────────────────────────────────────────────────────────────────────────
# CAU HINH
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
BATCH_SIZE   = 64
MAX_EPOCHS   = 30
VOCAB_SIZE   = 15000   # so tu toi da trong vocabulary
MAX_SEQ_LEN  = 100     # do dai toi da mot chuoi
EMBED_DIM    = 128     # embedding dimension
LSTM_UNITS   = 128     # BiLSTM units (moi huong)
CNN_FILTERS  = 64      # so filters TextCNN
CNN_KERNELS  = [2, 3, 4]  # kernel sizes TextCNN

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))  # .../bai2/
TUAN1_DIR     = os.path.dirname(BASE_DIR)                   # .../Tuan1/
PROCESSED_DIR = os.path.join(TUAN1_DIR, "results", "processed_data")
RESULTS_DIR   = os.path.join(TUAN1_DIR, "results", "bai2")
BAI1_CSV      = os.path.join(TUAN1_DIR, "results", "bai1", "model_results.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"TensorFlow version : {tf.__version__}")
print(f"GPU available      : {len(tf.config.list_physical_devices('GPU')) > 0}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DU LIEU DA TIEN XU LY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("1. LOAD DU LIEU DA TIEN XU LY (results/processed_data/)")
print("=" * 70)

df_train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
df_val   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
df_test  = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

print(f"  Train (goc + augmented) : {len(df_train):>6} mau")
print(f"  Validation (goc)        : {len(df_val):>6} mau")
print(f"  Test       (goc)        : {len(df_test):>6} mau")

texts_train = df_train['text'].fillna('').tolist()
texts_val   = df_val['text'].fillna('').tolist()
texts_test  = df_test['text'].fillna('').tolist()

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(df_train['polarity'])
class_names = label_encoder.classes_
n_classes   = len(class_names)

y_train = label_encoder.transform(df_train['polarity'])
y_val   = label_encoder.transform(df_val['polarity'])
y_test  = label_encoder.transform(df_test['polarity'])

# One-hot encode cho DL models
y_train_oh = keras.utils.to_categorical(y_train, n_classes)
y_val_oh   = keras.utils.to_categorical(y_val, n_classes)

# Class weights (xu ly mat can bang du lieu)
class_weights_arr = compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weights = dict(enumerate(class_weights_arr))
print(f"\n  Nhan    : {class_names.tolist()}")
print(f"  Weights : { {k: round(v, 3) for k, v in class_weights.items()} }")
print(f"\n  Phan bo nhan — Test:")
print("  " + df_test['polarity'].value_counts().to_string().replace('\n', '\n  '))

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. FEATURE ENGINEERING")
print("=" * 70)

# ── 2a. Word Tokenization + Padding (cho BiLSTM, TextCNN) ────────────────────
print("\n  [2a] Word Tokenization + Padding cho sequence models...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(texts_train)          # fit CHI tren train
vocab_actual = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)

def to_seq(texts):
    return pad_sequences(
        tokenizer.texts_to_sequences(texts),
        maxlen=MAX_SEQ_LEN, padding='post', truncating='post'
    )

X_train_seq = to_seq(texts_train)
X_val_seq   = to_seq(texts_val)
X_test_seq  = to_seq(texts_test)

print(f"     Vocab size         : {vocab_actual}")
print(f"     Max sequence length: {MAX_SEQ_LEN}")
print(f"     X_train_seq shape  : {X_train_seq.shape}")
print(f"     X_test_seq  shape  : {X_test_seq.shape}")

# ── 2b. TF-IDF (Unigram + Bigram, cho MLP) ───────────────────────────────────
print("\n  [2b] TF-IDF Vectorization (Unigram + Bigram) cho MLP...")
tfidf = TfidfVectorizer(
    max_features=8000, ngram_range=(1, 2),
    min_df=2, max_df=0.95, sublinear_tf=True
)
X_train_tfidf = tfidf.fit_transform(texts_train).toarray().astype(np.float32)
X_val_tfidf   = tfidf.transform(texts_val).toarray().astype(np.float32)
X_test_tfidf  = tfidf.transform(texts_test).toarray().astype(np.float32)
print(f"     TF-IDF features    : {X_train_tfidf.shape[1]}")

# ── 2c. Sentiment Lexicon Features (hand-crafted) ────────────────────────────
print("\n  [2c] Sentiment Lexicon Features (hand-crafted)...")

POS_WORDS = {
    'great','good','excellent','amazing','wonderful','fantastic','superb',
    'perfect','delicious','tasty','fresh','friendly','helpful','nice','love',
    'loved','best','outstanding','brilliant','enjoyable','pleasure','recommend',
    'recommended','satisfied','impressive','quality','pleasant','cozy','warm',
    'comfortable','scrumptious','flavorful','yummy','fabulous','terrific',
    'exceptional','attentive','generous','affordable','reasonable','worth',
    'clean','quick','fast','authentic','fresh','tender','juicy','crispy',
}
NEG_WORDS = {
    'bad','terrible','awful','horrible','poor','worst','disgusting','mediocre',
    'slow','cold','rude','unfriendly','expensive','overpriced','disappointed',
    'disappointing','bland','tasteless','overcooked','undercooked','noisy',
    'dirty','greasy','salty','dry','tough','stale','small','tiny','cramped',
    'waited','wait','long','never','not','no','nothing','wrong','problem',
    'unfortunately','avoid','horrible','lacking','boring','rushed','ignored',
    'fake','generic','average','mediocre','inedible','flavorless','soggy',
}
NEG_WORD_SET = {'not','no','never','neither','nor','without','lack','lacking'}

def lexicon_features(texts):
    feats = []
    for text in texts:
        words = str(text).lower().split()
        n = max(len(words), 1)
        n_pos   = sum(1 for w in words if w in POS_WORDS)
        n_neg   = sum(1 for w in words if w in NEG_WORDS)
        n_negate= sum(1 for w in words if w in NEG_WORD_SET)
        feats.append([
            n_pos, n_neg, n_negate,
            n_pos / n, n_neg / n,
            (n_pos - n_neg) / n,    # net sentiment score
            n_negate / n,           # negation density
            n,                      # doc length
        ])
    return np.array(feats, dtype=np.float32)

X_train_lex = lexicon_features(texts_train)
X_val_lex   = lexicon_features(texts_val)
X_test_lex  = lexicon_features(texts_test)
print(f"     Lexicon features   : {X_train_lex.shape[1]}")

# ── 2d. Ket hop TF-IDF + Lexicon (MLP input) ─────────────────────────────────
X_train_mlp = np.hstack([X_train_tfidf, X_train_lex])
X_val_mlp   = np.hstack([X_val_tfidf,   X_val_lex])
X_test_mlp  = np.hstack([X_test_tfidf,  X_test_lex])
print(f"\n     MLP input (TF-IDF + Lexicon): {X_train_mlp.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. HAM TIEN ICH
# ─────────────────────────────────────────────────────────────────────────────
def get_metrics(name, y_true, y_pred, y_score=None):
    """Tinh tat ca metrics cho mot mo hinh."""
    acc    = accuracy_score(y_true, y_pred)
    prec_m = precision_score(y_true, y_pred, average='macro',    zero_division=0)
    rec_m  = recall_score(y_true, y_pred,    average='macro',    zero_division=0)
    f1_m   = f1_score(y_true, y_pred,        average='macro',    zero_division=0)
    prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_w  = recall_score(y_true, y_pred,    average='weighted', zero_division=0)
    f1_w   = f1_score(y_true, y_pred,        average='weighted', zero_division=0)
    roc = float('nan')
    if y_score is not None:
        try:
            y_bin = label_binarize(y_true, classes=range(n_classes))
            roc   = roc_auc_score(y_bin, y_score, average='macro', multi_class='ovr')
        except Exception:
            pass
    return {
        'Model': name,
        'Accuracy':              round(acc,    4),
        'Precision (Macro)':     round(prec_m, 4),
        'Recall (Macro)':        round(rec_m,  4),
        'F1 (Macro)':            round(f1_m,   4),
        'Precision (Weighted)':  round(prec_w, 4),
        'Recall (Weighted)':     round(rec_w,  4),
        'F1 (Weighted)':         round(f1_w,   4),
        'ROC-AUC (Macro)':       round(roc, 4) if not np.isnan(roc) else float('nan'),
    }

CALLBACKS = [
    EarlyStopping(monitor='val_loss', patience=5,
                  restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-5, verbose=0),
]

dl_results   = {}   # {name: metrics dict}
dl_histories = {}   # {name: history dict}
dl_y_scores  = {}   # {name: softmax proba on test}
dl_y_preds   = {}   # {name: predicted labels on test}

# ─────────────────────────────────────────────────────────────────────────────
# 4. XAY DUNG VA HUAN LUYEN CAC MO HINH DL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. XAY DUNG VA HUAN LUYEN CAC MO HINH DL")
print("=" * 70)

# ────────────────────────────────────────────────────────────────────────────
# 4a. MLP — Multi-Layer Perceptron (TF-IDF + Lexicon features)
# ────────────────────────────────────────────────────────────────────────────
print("\n--- [4a] MLP (TF-IDF + Lexicon Features) ---")

inp = Input(shape=(X_train_mlp.shape[1],), name='mlp_in')
x   = layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(inp)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.4)(x)
x   = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.3)(x)
x   = layers.Dense(128, activation='relu')(x)
x   = layers.Dropout(0.2)(x)
out = layers.Dense(n_classes, activation='softmax')(x)

mlp_model = Model(inp, out, name='MLP')
mlp_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss='categorical_crossentropy', metrics=['accuracy'])
mlp_model.summary(print_fn=lambda s: None)   # quiet

hist = mlp_model.fit(
    X_train_mlp, y_train_oh,
    validation_data=(X_val_mlp, y_val_oh),
    batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
    class_weight=class_weights,
    callbacks=CALLBACKS, verbose=1
)
dl_histories['MLP'] = hist.history

mlp_proba = mlp_model.predict(X_test_mlp, verbose=0)
mlp_pred  = np.argmax(mlp_proba, axis=1)
dl_y_scores['MLP'] = mlp_proba
dl_y_preds['MLP']  = mlp_pred
dl_results['MLP']  = get_metrics('MLP', y_test, mlp_pred, mlp_proba)
r = dl_results['MLP']
print(f"  Test: Acc={r['Accuracy']:.4f}  F1={r['F1 (Macro)']:.4f}  ROC-AUC={r['ROC-AUC (Macro)']:.4f}")

# ────────────────────────────────────────────────────────────────────────────
# 4b. BiLSTM — Bidirectional LSTM voi trainable Embeddings
# ────────────────────────────────────────────────────────────────────────────
print("\n--- [4b] BiLSTM (Bidirectional LSTM + trainable Embedding) ---")

seq_in = Input(shape=(MAX_SEQ_LEN,), name='bilstm_in')
emb    = layers.Embedding(vocab_actual, EMBED_DIM, mask_zero=True)(seq_in)
emb    = layers.SpatialDropout1D(0.3)(emb)
bi1    = layers.Bidirectional(
    layers.LSTM(LSTM_UNITS, return_sequences=True,
                dropout=0.2, recurrent_dropout=0.2)
)(emb)
bi2    = layers.Bidirectional(
    layers.LSTM(LSTM_UNITS // 2, dropout=0.2)
)(bi1)
x      = layers.Dense(128, activation='relu')(bi2)
x      = layers.Dropout(0.3)(x)
out    = layers.Dense(n_classes, activation='softmax')(x)

bilstm_model = Model(seq_in, out, name='BiLSTM')
bilstm_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss='categorical_crossentropy', metrics=['accuracy'])

hist = bilstm_model.fit(
    X_train_seq, y_train_oh,
    validation_data=(X_val_seq, y_val_oh),
    batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
    class_weight=class_weights,
    callbacks=CALLBACKS, verbose=1
)
dl_histories['BiLSTM'] = hist.history

bilstm_proba = bilstm_model.predict(X_test_seq, verbose=0)
bilstm_pred  = np.argmax(bilstm_proba, axis=1)
dl_y_scores['BiLSTM'] = bilstm_proba
dl_y_preds['BiLSTM']  = bilstm_pred
dl_results['BiLSTM']  = get_metrics('BiLSTM', y_test, bilstm_pred, bilstm_proba)
r = dl_results['BiLSTM']
print(f"  Test: Acc={r['Accuracy']:.4f}  F1={r['F1 (Macro)']:.4f}  ROC-AUC={r['ROC-AUC (Macro)']:.4f}")

# ────────────────────────────────────────────────────────────────────────────
# 4c. TextCNN — Parallel Conv1D (Kim 2014)
# ────────────────────────────────────────────────────────────────────────────
print("\n--- [4c] TextCNN (Parallel Conv1D, kernel sizes={CNN_KERNELS}) ---")

seq_in2 = Input(shape=(MAX_SEQ_LEN,), name='cnn_in')
emb2    = layers.Embedding(vocab_actual, EMBED_DIM)(seq_in2)
emb2    = layers.SpatialDropout1D(0.3)(emb2)

conv_pool = []
for k in CNN_KERNELS:
    c = layers.Conv1D(CNN_FILTERS, k, activation='relu', padding='same')(emb2)
    c = layers.GlobalMaxPooling1D()(c)
    conv_pool.append(c)

concat = layers.Concatenate()(conv_pool)
x      = layers.Dense(128, activation='relu')(concat)
x      = layers.Dropout(0.4)(x)
x      = layers.Dense(64, activation='relu')(x)
x      = layers.Dropout(0.3)(x)
out    = layers.Dense(n_classes, activation='softmax')(x)

cnn_model = Model(seq_in2, out, name='TextCNN')
cnn_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss='categorical_crossentropy', metrics=['accuracy'])

hist = cnn_model.fit(
    X_train_seq, y_train_oh,
    validation_data=(X_val_seq, y_val_oh),
    batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
    class_weight=class_weights,
    callbacks=CALLBACKS, verbose=1
)
dl_histories['TextCNN'] = hist.history

cnn_proba = cnn_model.predict(X_test_seq, verbose=0)
cnn_pred  = np.argmax(cnn_proba, axis=1)
dl_y_scores['TextCNN'] = cnn_proba
dl_y_preds['TextCNN']  = cnn_pred
dl_results['TextCNN']  = get_metrics('TextCNN', y_test, cnn_pred, cnn_proba)
r = dl_results['TextCNN']
print(f"  Test: Acc={r['Accuracy']:.4f}  F1={r['F1 (Macro)']:.4f}  ROC-AUC={r['ROC-AUC (Macro)']:.4f}")

# ────────────────────────────────────────────────────────────────────────────
# 4d. BiLSTM + Self-Attention
# ────────────────────────────────────────────────────────────────────────────
print("\n--- [4d] BiLSTM + Self-Attention ---")

class SelfAttention(layers.Layer):
    """Bahdanau-style self-attention layer."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1,     use_bias=False)

    def call(self, hidden):                                  # (batch, seq, dim)
        score   = self.V(tf.nn.tanh(self.W(hidden)))         # (batch, seq, 1)
        weights = tf.nn.softmax(score, axis=1)               # (batch, seq, 1)
        context = tf.reduce_sum(weights * hidden, axis=1)    # (batch, dim)
        return context

    def get_config(self):
        cfg = super().get_config()
        return cfg

seq_in3 = Input(shape=(MAX_SEQ_LEN,), name='attn_in')
emb3    = layers.Embedding(vocab_actual, EMBED_DIM, mask_zero=True)(seq_in3)
emb3    = layers.SpatialDropout1D(0.3)(emb3)
bi_seq  = layers.Bidirectional(
    layers.LSTM(LSTM_UNITS, return_sequences=True,
                dropout=0.2, recurrent_dropout=0.2)
)(emb3)
ctx     = SelfAttention(LSTM_UNITS * 2, name='self_attn')(bi_seq)
x       = layers.Dense(128, activation='relu')(ctx)
x       = layers.Dropout(0.3)(x)
out     = layers.Dense(n_classes, activation='softmax')(x)

attn_model = Model(seq_in3, out, name='BiLSTM_Attention')
attn_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                    loss='categorical_crossentropy', metrics=['accuracy'])

hist = attn_model.fit(
    X_train_seq, y_train_oh,
    validation_data=(X_val_seq, y_val_oh),
    batch_size=BATCH_SIZE, epochs=MAX_EPOCHS,
    class_weight=class_weights,
    callbacks=CALLBACKS, verbose=1
)
dl_histories['BiLSTM+Attention'] = hist.history

attn_proba = attn_model.predict(X_test_seq, verbose=0)
attn_pred  = np.argmax(attn_proba, axis=1)
dl_y_scores['BiLSTM+Attention'] = attn_proba
dl_y_preds['BiLSTM+Attention']  = attn_pred
dl_results['BiLSTM+Attention']  = get_metrics('BiLSTM+Attention', y_test, attn_pred, attn_proba)
r = dl_results['BiLSTM+Attention']
print(f"  Test: Acc={r['Accuracy']:.4f}  F1={r['F1 (Macro)']:.4f}  ROC-AUC={r['ROC-AUC (Macro)']:.4f}")

# ────────────────────────────────────────────────────────────────────────────
# 4e. DL Ensemble — Soft Voting (BiLSTM + TextCNN + BiLSTM+Attention)
# ────────────────────────────────────────────────────────────────────────────
print("\n--- [4e] DL Ensemble: Soft Voting (BiLSTM + TextCNN + BiLSTM+Attention) ---")
ens_proba = (bilstm_proba + cnn_proba + attn_proba) / 3.0
ens_pred  = np.argmax(ens_proba, axis=1)
dl_y_scores['DL Ensemble'] = ens_proba
dl_y_preds['DL Ensemble']  = ens_pred
dl_results['DL Ensemble']  = get_metrics('DL Ensemble', y_test, ens_pred, ens_proba)
r = dl_results['DL Ensemble']
print(f"  Test: Acc={r['Accuracy']:.4f}  F1={r['F1 (Macro)']:.4f}  ROC-AUC={r['ROC-AUC (Macro)']:.4f}")
print(f"\n  Classification Report — DL Ensemble (Test):")
print(classification_report(y_test, ens_pred, target_names=class_names, zero_division=0))

# ─────────────────────────────────────────────────────────────────────────────
# 5. BANG KET QUA DL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. BANG KET QUA DL MODELS (Test Set)")
print("=" * 70)

metrics_cols = [
    'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 (Macro)',
    'Precision (Weighted)', 'Recall (Weighted)', 'F1 (Weighted)', 'ROC-AUC (Macro)'
]
df_dl = pd.DataFrame([dl_results[n] for n in dl_results]).set_index('Model')
print("\n" + df_dl[metrics_cols].to_string())
df_dl.to_csv(os.path.join(RESULTS_DIR, "dl_results.csv"))
print(f"\n[OK] Da luu -> results/bai2/dl_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SO SANH VOI BAI1 ML BASELINES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. SO SANH VOI BAI1 ML BASELINES")
print("=" * 70)

compare_cols = ['Accuracy', 'F1 (Macro)', 'Precision (Macro)',
                'Recall (Macro)', 'ROC-AUC (Macro)']

df_compare_dl = df_dl[compare_cols].copy()
df_compare_dl['Type'] = 'Deep Learning'

if os.path.exists(BAI1_CSV):
    df_bai1 = pd.read_csv(BAI1_CSV, index_col='Model')
    common   = [c for c in compare_cols if c in df_bai1.columns]
    df_bai1_cmp = df_bai1[common].copy()
    df_bai1_cmp['Type'] = 'ML Baseline'
    df_compare = pd.concat([df_compare_dl, df_bai1_cmp])
    print("\nBANG SO SANH TOAN BO (DL + ML Baseline):")
    print(df_compare[compare_cols + ['Type']].to_string())
else:
    df_compare = df_compare_dl
    print(f"\n[WARNING] Khong tim thay {BAI1_CSV}")
    print("Chi hien thi ket qua DL.\n")
    print(df_compare.to_string())

df_compare.to_csv(os.path.join(RESULTS_DIR, "full_comparison.csv"))
print(f"\n[OK] Da luu -> results/bai2/full_comparison.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 7. TRUC QUAN HOA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. TRUC QUAN HOA")
print("=" * 70)

DL_COLORS = {
    'MLP':               '#4A90D9',
    'BiLSTM':            '#50B86C',
    'TextCNN':           '#F5A623',
    'BiLSTM+Attention':  '#9B59B6',
    'DL Ensemble':       '#E74C6F',
}
ML_COLORS_LIST = ['#7F8C8D', '#95A5A6', '#BDC3C7', '#D5DBDB']

# ── 7.1 Training History ─────────────────────────────────────────────────────
n_hist = len(dl_histories)
fig, axes = plt.subplots(2, n_hist, figsize=(5 * n_hist, 8))
fig.suptitle("Training History — DL Models\n(SemEval 2014 Restaurant ABSA)",
             fontsize=14, fontweight='bold')

for col, (mname, hist) in enumerate(dl_histories.items()):
    epochs = range(1, len(hist['loss']) + 1)
    ax_l = axes[0, col]
    ax_a = axes[1, col]

    ax_l.plot(epochs, hist['loss'],     color='#4A90D9', linewidth=2, label='Train')
    ax_l.plot(epochs, hist['val_loss'], color='#E74C6F', linewidth=2, linestyle='--', label='Val')
    ax_l.set_title(mname, fontsize=11, fontweight='bold')
    ax_l.set_ylabel('Loss')
    ax_l.legend(fontsize=9)
    ax_l.grid(alpha=0.3)

    ax_a.plot(epochs, hist['accuracy'],     color='#4A90D9', linewidth=2, label='Train')
    ax_a.plot(epochs, hist['val_accuracy'], color='#E74C6F', linewidth=2, linestyle='--', label='Val')
    ax_a.set_ylabel('Accuracy')
    ax_a.set_xlabel('Epoch')
    ax_a.legend(fontsize=9)
    ax_a.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "training_history.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] training_history.png")

# ── 7.2 Comparison Bar Chart (DL + ML) ───────────────────────────────────────
all_model_names = list(dl_results.keys())
all_colors_list = [DL_COLORS.get(m, '#999') for m in all_model_names]
all_vals = {m: [dl_results[n].get(m, 0) for n in all_model_names] for m in compare_cols}

if os.path.exists(BAI1_CSV):
    df_bai1_all = pd.read_csv(BAI1_CSV, index_col='Model')
    bai1_names  = df_bai1_all.index.tolist()
    all_model_names = all_model_names + bai1_names
    all_colors_list = all_colors_list + ML_COLORS_LIST[:len(bai1_names)]
    for m in compare_cols:
        ml_v = [float(df_bai1_all.loc[bm, m]) if m in df_bai1_all.columns else 0
                for bm in bai1_names]
        all_vals[m] = all_vals[m] + ml_v

n_dl = len(dl_results)
fig, axes = plt.subplots(1, len(compare_cols), figsize=(4.5 * len(compare_cols), 8))
fig.suptitle("So sanh toan bo mo hinh — DL vs ML Baseline\n(SemEval 2014 Restaurant ABSA)",
             fontsize=14, fontweight='bold')

for ax, metric in zip(axes, compare_cols):
    vals = all_vals[metric]
    bars = ax.barh(all_model_names, vals,
                   color=all_colors_list, edgecolor='white', linewidth=0.8)
    ax.set_xlim(0, 1.18)
    ax.set_title(metric, fontsize=10, fontweight='bold')
    ax.set_xlabel('Score')
    if os.path.exists(BAI1_CSV):
        ax.axhline(y=n_dl - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(0.01, n_dl - 0.3, 'ML', fontsize=8, color='gray')
        ax.text(0.01, n_dl + 0.1, 'DL', fontsize=8, color='gray')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "model_comparison_dl_vs_ml.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] model_comparison_dl_vs_ml.png")

# ── 7.3 Confusion Matrices (DL models) ───────────────────────────────────────
n_models = len(dl_y_preds)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
fig.suptitle("Confusion Matrix — DL Models (Test Set)\n(SemEval 2014 Restaurant ABSA)",
             fontsize=13, fontweight='bold')

for ax, (mname, pred) in zip(axes, dl_y_preds.items()):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(mname, fontsize=10, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "confusion_matrices_dl.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] confusion_matrices_dl.png")

# ── 7.4 ROC Curves (DL models) ───────────────────────────────────────────────
y_test_bin = label_binarize(y_test, classes=range(n_classes))
fig, axes  = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
fig.suptitle("ROC Curves (One-vs-Rest) — DL Models (Test Set)\n(SemEval 2014 Restaurant ABSA)",
             fontsize=13, fontweight='bold')

dl_color_vals = list(DL_COLORS.values())
for ci in range(n_classes):
    ax = axes[ci]
    ax.set_title(f"Class: {class_names[ci]}", fontsize=11, fontweight='bold')
    for i, (mname, score) in enumerate(dl_y_scores.items()):
        if score.ndim < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_test_bin[:, ci], score[:, ci])
            roc_val     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=dl_color_vals[i % len(dl_color_vals)],
                    linewidth=2, label=f"{mname} (AUC={roc_val:.3f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
out = os.path.join(RESULTS_DIR, "roc_curves_dl.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] roc_curves_dl.png")

# ── 7.5 Best Validation F1 per model (bar) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
val_f1s  = [max(h.get('val_accuracy', [0])) for h in dl_histories.values()]
hnames   = list(dl_histories.keys())
hcolors  = [DL_COLORS.get(n, '#999') for n in hnames]
bars = ax.bar(hnames, val_f1s, color=hcolors, edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, val_f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_title("Best Validation Accuracy per DL Model\n(SemEval 2014 Restaurant ABSA)",
             fontsize=13, fontweight='bold')
ax.set_ylabel("Best Val Accuracy")
ax.set_ylim(0, 1.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
out = os.path.join(RESULTS_DIR, "val_accuracy_dl.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  [OK] val_accuracy_dl.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. KET LUAN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. KET LUAN")
print("=" * 70)

key_metrics = ['Accuracy', 'F1 (Macro)', 'Precision (Macro)',
               'Recall (Macro)', 'ROC-AUC (Macro)']
print("\nBest DL Model theo tung metric (tren Test):")
for m in key_metrics:
    if m in df_dl.columns:
        best     = df_dl[m].idxmax()
        best_val = df_dl[m].max()
        print(f"  {m:25s}: {best} ({best_val:.4f})")

print(f"""
  +--------------------------------------------------------------+
  |           KET QUA BAI 2 — DEEP LEARNING                     |
  +--------------------------------------------------------------+
  |  Mo hinh DL duoc xay dung:                                  |
  |    1. MLP                (TF-IDF + Lexicon features)        |
  |    2. BiLSTM             (Trainable Word Embeddings)        |
  |    3. TextCNN            (Parallel Conv1D, Kim 2014)        |
  |    4. BiLSTM+Attention   (Self-Attention Mechanism)         |
  |    5. DL Ensemble        (Soft Voting: 2+3+4)               |
  +--------------------------------------------------------------+
  |  Feature Engineering:                                        |
  |    - Word Tokenization + Padding (cho 2,3,4,5)              |
  |    - TF-IDF Unigram+Bigram (cho 1)                          |
  |    - Sentiment Lexicon 8 features (cho 1)                   |
  |    - Class weights de xu ly mat can bang                    |
  +--------------------------------------------------------------+
  |  Files da luu trong results/bai2/:                          |
  |    dl_results.csv                                           |
  |    full_comparison.csv                                      |
  |    training_history.png                                     |
  |    model_comparison_dl_vs_ml.png                            |
  |    confusion_matrices_dl.png                                |
  |    roc_curves_dl.png                                        |
  |    val_accuracy_dl.png                                      |
  +--------------------------------------------------------------+
""")
print("=" * 70)
print("HOAN TAT BAI 2 — DEEP LEARNING SENTIMENT ANALYSIS!")
print("=" * 70)
