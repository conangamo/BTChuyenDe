# -*- coding: utf-8 -*-
# =============================================================================
# Tien xu ly & Lam giau du lieu - SemEval 2014 ABSA Dataset
# Buoc 1 : Chia Train / Validation / Test
# Buoc 2 : Lam sach du lieu
# Buoc 3 : Data Augmentation (CHI tren tap Train)
# Tap Validation va Test GIU NGUYEN dang goc
# =============================================================================

import os
import sys
import re
import random
import warnings
import unicodedata

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CAU HINH
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
VAL_RATIO     = 0.15   # 15% cua toan bo du lieu dung lam Validation
TEST_RATIO    = 0.15   # 15% cua toan bo du lieu dung lam Test
# Con lai 70% dung lam Train

AUG_PER_SAMPLE = 2     # So ban augmented sinh ra cho moi mau Train goc

DATA_DIR = os.path.join(
    "datasets", "charitarth",
    "semeval-2014-task-4-aspectbasedsentimentanalysis",
    "versions", "4"
)
RESULTS_DIR   = "results"
PROCESSED_DIR = os.path.join(RESULTS_DIR, "processed_data")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DOC DU LIEU GOC
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("BUOC 1 — DOC DU LIEU GOC")
print("=" * 70)

train_file = os.path.join(DATA_DIR, "Restaurants_Train_v2.csv")
df_raw = pd.read_csv(train_file)

print(f"  File         : {train_file}")
print(f"  So hang     : {len(df_raw)}")
print(f"  Cac cot     : {df_raw.columns.tolist()}")
print(f"\n  Phan bo nhan goc:")
print(df_raw['polarity'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2. LAM SACH DU LIEU
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BUOC 2 — LAM SACH DU LIEU")
print("=" * 70)

df = df_raw.copy()

# 2.1  Loai bo hang thieu du lieu quan trong
before = len(df)
df = df.dropna(subset=['Sentence', 'Aspect Term', 'polarity'])
print(f"  Loai bo NaN          : {before - len(df)} hang -> con {len(df)} hang")

# 2.2  Loai nhan 'conflict' (rat it, gay nhieu)
before = len(df)
df = df[df['polarity'].isin(['positive', 'negative', 'neutral'])]
print(f"  Loai nhan 'conflict' : {before - len(df)} hang -> con {len(df)} hang")

# 2.3  Loai bo hang trung lap hoan toan
before = len(df)
df = df.drop_duplicates(subset=['Sentence', 'Aspect Term', 'polarity'])
print(f"  Loai trung lap       : {before - len(df)} hang -> con {len(df)} hang")

# 2.4  Ham lam sach van ban
def clean_text(text):
    """
    Lam sach va chuan hoa van ban:
      - Giai ma unicode ve dang NFC
      - Loai bo ky tu dieu khien / khong in duoc
      - Rut gon khoang trang thua
      - Lowercase
      - Loai bo URL, email
      - Loai bo HTML entities & tags
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+\.\S+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def clean_aspect(text):
    """Lam sach nhe aspect term (giu lowercase nhung khong loai dau cau)."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

df['Sentence_clean']    = df['Sentence'].apply(clean_text)
df['Aspect_clean']      = df['Aspect Term'].apply(clean_aspect)

# 2.5  Tao cot feature text = sentence + [SEP] + aspect
df['text'] = df['Sentence_clean'] + ' [SEP] ' + df['Aspect_clean']

# 2.6  Loai bo cac hang ma text qua ngan sau khi lam sach
before = len(df)
df = df[df['Sentence_clean'].str.len() >= 5]
print(f"  Loai van ban qua ngan: {before - len(df)} hang -> con {len(df)} hang")

print(f"\n  Phan bo nhan sau lam sach:")
print(df['polarity'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. CHIA TRAIN / VALIDATION / TEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BUOC 3 — CHIA TRAIN / VALIDATION / TEST")
print(f"         Ti le: Train {100-int((VAL_RATIO+TEST_RATIO)*100)}% "
      f"| Val {int(VAL_RATIO*100)}% | Test {int(TEST_RATIO*100)}%")
print("=" * 70)

# Chia lan 1: tach phan Test
df_trainval, df_test = train_test_split(
    df,
    test_size=TEST_RATIO,
    random_state=RANDOM_SEED,
    stratify=df['polarity']
)

# Chia lan 2: tu phan con lai tach Validation
val_ratio_adjusted = VAL_RATIO / (1 - TEST_RATIO)
df_train_raw, df_val = train_test_split(
    df_trainval,
    test_size=val_ratio_adjusted,
    random_state=RANDOM_SEED,
    stratify=df_trainval['polarity']
)

print(f"  Train (truoc augmentation) : {len(df_train_raw):>5} mau")
print(f"  Validation                 : {len(df_val):>5} mau")
print(f"  Test                       : {len(df_test):>5} mau")
print(f"  Tong cong                  : {len(df_train_raw)+len(df_val)+len(df_test):>5} mau")

print(f"\n  Phan bo nhan — Train raw:")
print("  " + df_train_raw['polarity'].value_counts().to_string().replace('\n', '\n  '))
print(f"\n  Phan bo nhan — Validation (GIU NGUYEN goc):")
print("  " + df_val['polarity'].value_counts().to_string().replace('\n', '\n  '))
print(f"\n  Phan bo nhan — Test (GIU NGUYEN goc):")
print("  " + df_test['polarity'].value_counts().to_string().replace('\n', '\n  '))

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA AUGMENTATION — CHI AP DUNG TREN TAP TRAIN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BUOC 4 — DATA AUGMENTATION (CHI TREN TAP TRAIN)")
print("=" * 70)
print("  [!] Validation va Test KHONG bi thay doi.\n")

# ── 4a. Tu dong nghia theo domain nha hang ────────────────────────────────────
SYNONYM_MAP = {
    # thuc an
    "food": ["dish", "meal", "cuisine", "fare"],
    "meal": ["food", "dish", "plate"],
    "dish": ["food", "meal", "cuisine"],
    "cuisine": ["food", "cooking", "dishes"],
    # dich vu
    "service": ["staff service", "hospitality", "customer care"],
    "staff": ["employees", "workers", "personnel", "team"],
    "waiter": ["server", "waitstaff", "attendant"],
    "waiters": ["servers", "waitstaff", "attendants"],
    "waitress": ["server", "waitstaff"],
    "manager": ["supervisor", "head server"],
    # do an cu the
    "pizza": ["pie", "pizza dish"],
    "pasta": ["noodles", "pasta dish"],
    "steak": ["beef steak", "grilled steak"],
    "dessert": ["sweet", "pudding"],
    "drink": ["beverage", "refreshment"],
    "wine": ["vino", "red wine", "white wine"],
    "coffee": ["espresso", "java", "brew"],
    # khong gian
    "atmosphere": ["ambiance", "ambience", "vibe"],
    "ambiance": ["atmosphere", "ambience", "setting"],
    "place": ["spot", "venue", "location", "restaurant"],
    "restaurant": ["eatery", "venue", "diner"],
    "menu": ["offerings", "selection", "choices", "menu list"],
    "price": ["cost", "rate", "charge"],
    "prices": ["costs", "rates", "charges"],
    # tinh tu sentiment
    "great": ["excellent", "fantastic", "wonderful", "superb"],
    "good": ["nice", "decent", "fine", "solid"],
    "bad": ["poor", "terrible", "awful", "dreadful"],
    "delicious": ["tasty", "scrumptious", "flavorful", "yummy"],
    "fresh": ["crisp", "newly made"],
    "cold": ["chilled", "tepid", "lukewarm"],
    "hot": ["warm", "piping hot", "steaming"],
    "slow": ["sluggish", "unhurried"],
    "friendly": ["welcoming", "pleasant", "amiable", "courteous"],
    "rude": ["unfriendly", "impolite", "discourteous"],
    "expensive": ["costly", "overpriced", "pricey"],
    "cheap": ["affordable", "inexpensive", "budget-friendly"],
    "average": ["mediocre", "ordinary", "so-so"],
    "small": ["tiny", "cramped", "compact"],
    "large": ["big", "spacious", "generous"],
    "long": ["lengthy", "extended"],
    "clean": ["tidy", "spotless", "neat"],
    "dirty": ["messy", "unclean", "grimy"],
    "comfortable": ["cozy", "pleasant", "relaxing"],
    "noisy": ["loud", "boisterous", "rowdy"],
    "quiet": ["peaceful", "calm", "serene"],
    "recommended": ["suggested", "highly rated"],
    "overcooked": ["burnt", "overdone"],
    "undercooked": ["raw", "underdone"],
    "bland": ["tasteless", "flavorless", "insipid"],
    "salty": ["over-salted", "too salty"],
    "sweet": ["sugary", "dessert-like"],
}

def synonym_replacement(sentence, n=2):
    """Thay the ngau nhien n tu bang tu dong nghia."""
    words = sentence.split()
    new_words = words.copy()
    replaceable_idx = [i for i, w in enumerate(words) if w in SYNONYM_MAP]
    random.shuffle(replaceable_idx)
    n_replaced = 0
    for idx in replaceable_idx:
        if n_replaced >= n:
            break
        synonyms = SYNONYM_MAP[new_words[idx]]
        if synonyms:
            new_words[idx] = random.choice(synonyms)
            n_replaced += 1
    return ' '.join(new_words)

# ── 4b. Random Deletion ───────────────────────────────────────────────────────
def random_deletion(sentence, p=0.10):
    """Xoa ngau nhien moi tu voi xac suat p."""
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = [w for w in words if random.random() > p]
    return ' '.join(new_words) if new_words else sentence

# ── 4c. Random Swap ───────────────────────────────────────────────────────────
def random_swap(sentence, n=1):
    """Hoan doi ngau nhien n cap tu trong cau."""
    words = sentence.split()
    if len(words) < 2:
        return sentence
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return ' '.join(new_words)

# ── 4d. Pseudo Back-Translation (paraphrasing rule-based) ────────────────────
NEGATION_PAIRS = [
    ("not very", "somewhat"), ("not really", "barely"), ("n't", " not"),
    ("isn't", "is not"), ("wasn't", "was not"), ("won't", "will not"),
    ("can't", "cannot"), ("didn't", "did not"), ("doesn't", "does not"),
    ("couldn't", "could not"), ("wouldn't", "would not"),
]
PHRASE_REPHRASE = [
    (r"\bthe food\b", "the cuisine"),
    (r"\bwas (\w+)\b", r"seemed \1"),
    (r"\bwere (\w+)\b", r"appeared \1"),
    (r"\bi think\b", "i believe"),
    (r"\bi feel\b", "i believe"),
    (r"\breally\b", "truly"),
    (r"\bvery\b", "quite"),
    (r"\bso\b", "extremely"),
    (r"\bwe had\b", "we got"),
    (r"\bwe got\b", "we received"),
    (r"\bwould recommend\b", "highly suggest"),
    (r"\bgot\b", "received"),
    (r"\bpretty\b", "fairly"),
    (r"\bkind of\b", "somewhat"),
]

def pseudo_back_translation(sentence):
    """Paraphrase nhe cau bang cac cap bieu thuc thay the."""
    text = sentence
    for old, new in NEGATION_PAIRS:
        text = text.replace(old, new)
    rules = random.sample(PHRASE_REPHRASE, min(2, len(PHRASE_REPHRASE)))
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text)
    return text

# ── 4e. Aspect-Aware Augmentation ─────────────────────────────────────────────
def aspect_aware_augment(sentence, aspect):
    """
    Augmentation co nhan thuc ve aspect:
    - Chen them cum nhan manh aspect vao dau cau
    - Ket hop synonym replacement
    """
    prefix_templates = [
        f"regarding {aspect}, ",
        f"as for {aspect}, ",
        f"speaking of {aspect}, ",
        f"when it comes to {aspect}, ",
        f"in terms of {aspect}, ",
    ]
    result = synonym_replacement(sentence, n=1)
    if random.random() > 0.5:
        result = random.choice(prefix_templates) + result
    return result

# ── 4f. Tong hop: tao augmented data ─────────────────────────────────────────
AUG_TECHNIQUES = [
    ("synonym_replacement", lambda s, a: synonym_replacement(s, n=2)),
    ("random_deletion",     lambda s, a: random_deletion(s, p=0.10)),
    ("random_swap",         lambda s, a: random_swap(s, n=1)),
    ("pseudo_backtrans",    lambda s, a: pseudo_back_translation(s)),
    ("aspect_aware",        lambda s, a: aspect_aware_augment(s, a)),
]

def augment_sample(row, n=AUG_PER_SAMPLE):
    """Tao n ban augmented cho 1 mau."""
    sentence = row['Sentence_clean']
    aspect   = row['Aspect_clean']
    chosen   = random.sample(AUG_TECHNIQUES, min(n, len(AUG_TECHNIQUES)))
    augmented = []
    for tech_name, func in chosen:
        new_sentence = func(sentence, aspect)
        if new_sentence and new_sentence != sentence:
            new_text = new_sentence + ' [SEP] ' + aspect
            augmented.append({
                'id'             : row['id'],
                'Sentence'       : row['Sentence'],
                'Aspect Term'    : row['Aspect Term'],
                'polarity'       : row['polarity'],
                'from'           : row['from'],
                'to'             : row['to'],
                'Sentence_clean' : new_sentence,
                'Aspect_clean'   : aspect,
                'text'           : new_text,
                'aug_technique'  : tech_name,
                'is_augmented'   : True,
            })
    return augmented

print(f"  Dang sinh augmented data ({AUG_PER_SAMPLE} ban/mau, {len(AUG_TECHNIQUES)} ky thuat) ...")

aug_rows = []
for _, row in df_train_raw.iterrows():
    aug_rows.extend(augment_sample(row, n=AUG_PER_SAMPLE))

df_aug = pd.DataFrame(aug_rows)

# Them cot is_augmented cho du lieu goc
df_train_raw = df_train_raw.copy()
df_train_raw['aug_technique'] = 'original'
df_train_raw['is_augmented']  = False

# Gop du lieu goc + augmented
df_train = pd.concat([df_train_raw, df_aug], ignore_index=True)

print(f"\n  -- Ket qua sau Data Augmentation --")
print(f"  Train goc             : {len(df_train_raw):>6} mau")
print(f"  Train augmented them  : {len(df_aug):>6} mau")
print(f"  Train tong cong       : {len(df_train):>6} mau")
print(f"  Validation            : {len(df_val):>6} mau  (KHONG THAY DOI)")
print(f"  Test                  : {len(df_test):>6} mau  (KHONG THAY DOI)")

print(f"\n  Phan bo nhan — Train (sau augmentation):")
print("  " + df_train['polarity'].value_counts().to_string().replace('\n', '\n  '))

print(f"\n  Phan bo theo ky thuat augmentation:")
print("  " + df_train['aug_technique'].value_counts().to_string().replace('\n', '\n  '))

# ─────────────────────────────────────────────────────────────────────────────
# 5. LUU CAC TAP DU LIEU DA XU LY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BUOC 5 — LUU CAC TAP DU LIEU")
print("=" * 70)

SAVE_COLS = ['id', 'Sentence', 'Aspect Term', 'polarity',
             'from', 'to', 'Sentence_clean', 'Aspect_clean', 'text',
             'aug_technique', 'is_augmented']

train_path = os.path.join(PROCESSED_DIR, "train.csv")
val_path   = os.path.join(PROCESSED_DIR, "val.csv")
test_path  = os.path.join(PROCESSED_DIR, "test.csv")

df_val_save  = df_val.copy()
df_test_save = df_test.copy()
df_val_save['aug_technique']  = 'original'
df_val_save['is_augmented']   = False
df_test_save['aug_technique'] = 'original'
df_test_save['is_augmented']  = False

df_train[SAVE_COLS].to_csv(train_path, index=False, encoding='utf-8')
df_val_save[SAVE_COLS].to_csv(val_path, index=False, encoding='utf-8')
df_test_save[SAVE_COLS].to_csv(test_path, index=False, encoding='utf-8')

print(f"  Da luu: {train_path}  ({len(df_train)} hang)")
print(f"  Da luu: {val_path}    ({len(df_val_save)} hang)")
print(f"  Da luu: {test_path}   ({len(df_test_save)} hang)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. BIEU DO TRUC QUAN HOA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BUOC 6 — TRUC QUAN HOA")
print("=" * 70)

palette = {
    'positive': '#50B86C',
    'negative': '#E74C6F',
    'neutral' : '#4A90D9',
}

# ── 6.1 Phan bo nhan ba tap ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Phan bo nhan Sentiment — Train / Validation / Test\n"
             "(SemEval 2014 Restaurant ABSA)",
             fontsize=15, fontweight='bold')

datasets = [
    (df_train[df_train['is_augmented'] == False], "Train (goc, truoc aug)"),
    (df_val_save,  "Validation (GIU NGUYEN)"),
    (df_test_save, "Test (GIU NGUYEN)"),
]

for ax, (data, title) in zip(axes, datasets):
    counts = data['polarity'].value_counts()
    colors = [palette.get(lbl, '#999') for lbl in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.2)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel("So luong mau")
    ax.set_xlabel("Nhan Sentiment")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "split_label_distribution.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Da luu: {out_path}")

# ── 6.2 So sanh Train truoc va sau augmentation ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("So sanh Train: Truoc va Sau Data Augmentation\n"
             "(SemEval 2014 Restaurant ABSA)",
             fontsize=14, fontweight='bold')

for ax, (data, title) in zip(axes, [
    (df_train_raw, "Truoc Augmentation"),
    (df_train,     "Sau Augmentation (goc + aug)"),
]):
    counts = data['polarity'].value_counts()
    colors = [palette.get(lbl, '#999') for lbl in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.2)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel("So luong mau")
    ax.set_xlabel("Nhan Sentiment")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "augmentation_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Da luu: {out_path}")

# ── 6.3 Pie chart ky thuat augmentation ──────────────────────────────────────
aug_only = df_train[df_train['is_augmented'] == True]
aug_counts = aug_only['aug_technique'].value_counts()

fig, ax = plt.subplots(figsize=(9, 7))
wedge_colors = ['#4A90D9', '#50B86C', '#F5A623', '#E74C6F', '#9B59B6']
wedges, texts, autotexts = ax.pie(
    aug_counts.values,
    labels=aug_counts.index,
    colors=wedge_colors[:len(aug_counts)],
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.82,
    wedgeprops=dict(edgecolor='white', linewidth=1.5)
)
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight('bold')
ax.set_title("Ti le cac ky thuat Data Augmentation (tren tap Train)\n"
             "(SemEval 2014 Restaurant ABSA)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "augmentation_techniques.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Da luu: {out_path}")

# ── 6.4 Bar chart so mau ba tap ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
split_labels = ['Train\n(goc + aug)', 'Validation\n(goc)', 'Test\n(goc)']
split_counts = [len(df_train), len(df_val_save), len(df_test_save)]
bar_colors   = ['#4A90D9', '#50B86C', '#F5A623']
bars = ax.barh(split_labels, split_counts, color=bar_colors,
               edgecolor='white', linewidth=1.2, height=0.5)
for bar, val in zip(bars, split_counts):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
            f'{val:,}', va='center', fontweight='bold', fontsize=12)
ax.set_xlabel("So luong mau", fontsize=12)
ax.set_title("Tong so mau trong tung tap du lieu\n"
             "(SemEval 2014 Restaurant ABSA)",
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, max(split_counts) * 1.18)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "dataset_split_summary.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Da luu: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. BAO CAO TONG KET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BUOC 7 — BAO CAO TONG KET")
print("=" * 70)

total_all = len(df_train) + len(df_val_save) + len(df_test_save)

print(f"""
  +-------------------------------------------------------------+
  |           KET QUA TIEN XU LY DU LIEU                       |
  +-------------------------------------------------------------+
  |  Du lieu goc sau lam sach      : {len(df):>6} mau              |
  +--------------------+----+----------------------------------+
  |  TAP DU LIEU       |  So mau   |  Ghi chu                 |
  +--------------------+-----------+--------------------------+
  |  Train (tong)      |  {len(df_train):>6}   |  goc + augmented        |
  |    > Goc           |  {len(df_train_raw):>6}   |  ~70% data goc         |
  |    > Augmented     |  {len(df_aug):>6}   |  {AUG_PER_SAMPLE} ban/mau x {len(AUG_TECHNIQUES)} ky thuat    |
  |  Validation        |  {len(df_val_save):>6}   |  ~15% GIU NGUYEN       |
  |  Test              |  {len(df_test_save):>6}   |  ~15% GIU NGUYEN       |
  +--------------------+-----------+--------------------------+
  |  Tong (Train+Val+Test)         : {total_all:>6} mau              |
  +-------------------------------------------------------------+
  |  Ky thuat Data Augmentation duoc ap dung:                  |
  |    1. Synonym Replacement   (thay tu dong nghia)           |
  |    2. Random Deletion       (xoa tu ngau nhien)            |
  |    3. Random Swap           (hoan doi tu ngau nhien)       |
  |    4. Pseudo Back-Translation (paraphrasing rule-based)    |
  |    5. Aspect-Aware Aug.     (nhan manh aspect term)        |
  +-------------------------------------------------------------+
  |  Files da luu:                                             |
  |    results/processed_data/train.csv                        |
  |    results/processed_data/val.csv                          |
  |    results/processed_data/test.csv                         |
  |    results/split_label_distribution.png                    |
  |    results/augmentation_comparison.png                     |
  |    results/augmentation_techniques.png                     |
  |    results/dataset_split_summary.png                       |
  +-------------------------------------------------------------+
""")
print("=" * 70)
print("HOAN TAT TIEN XU LY DU LIEU!")
print("=" * 70)
