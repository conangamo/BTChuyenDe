# BÁO CÁO GIẢI TRÌNH DỰ ÁN ABSA — TUẦN 2
## Aspect-Based Sentiment Analysis trên SemEval 2014 Restaurant Dataset

> **Dataset:** SemEval 2014 Task 4 — Restaurant Reviews (Tiếng Anh)  
> **File chính:** `Tuan1/bai2/SemEval2014_DL.py`, `Tuan1/dataPreprocessing/data_preprocessing.py`, `Tuan1/bai1/SemEval2014Restaurant.py`  
> **Kết quả đã chạy và kiểm chứng:** `Tuan1/results/bai2/dl_results.csv`, `full_comparison.csv`

---

## Mục lục

1. [Build Model — Xây dựng mô hình SA sử dụng Deep Learning](#i-build-model)
2. [Evaluation Metric — Đánh giá mô hình](#ii-evaluation-metric)
3. [Compare Models — So sánh với baseline](#iii-compare-models)
4. [Data Augmentation & Làm sạch dữ liệu](#iv-data-augmentation--làm-sạch-dữ-liệu)
5. [Feature Engineering — Biểu diễn đặc trưng](#v-feature-engineering)
6. [Ensemble Methods — Kết hợp mô hình](#vi-ensemble-methods)
7. [Bảng tổng kết](#vii-bảng-tổng-kết)

---

## I. BUILD MODEL

### Phương pháp

Dự án xây dựng **4 mô hình Deep Learning độc lập** và **1 mô hình Ensemble**:

| Mô hình | Kiến trúc | Input |
|---|---|---|
| MLP | Dense 512→256→128→Softmax | TF-IDF (8000 chiều) + Lexicon (8 chiều) |
| BiLSTM | Embedding→BiLSTM×2→Dense→Softmax | Chuỗi từ tokenized (100 token) |
| TextCNN | Embedding→Conv1D(kernel 2,3,4)→GlobalMaxPool→Softmax | Chuỗi từ tokenized (100 token) |
| BiLSTM+Attention | Embedding→BiLSTM→SelfAttention→Dense→Softmax | Chuỗi từ tokenized (100 token) |
| DL Ensemble | Soft Voting trên xác suất Softmax của BiLSTM+TextCNN+Attention | Kết hợp 3 mô hình trên |

### Mục đích — Tại sao ABSA khác SA thông thường?

**SA thông thường** gán một nhãn cảm xúc chung cho toàn bộ câu.  
**ABSA (Aspect-Based SA)** xác định cảm xúc riêng cho **từng khía cạnh** được nhắc đến trong câu.

**Ví dụ minh họa:**

> *"The food was excellent but the service was terrible."*

| Khía cạnh (Aspect Term) | Nhãn cảm xúc (Polarity) |
|---|---|
| food | **positive** |
| service | **negative** |

SA thông thường chỉ cho ra 1 nhãn; ABSA cho ra 2 nhãn riêng biệt. Dataset có 3 nhãn: `positive`, `negative`, `neutral`.

**Đầu vào mô hình:** chuỗi `câu [SEP] aspect` (ví dụ: `"the food was excellent [SEP] food"`)  
**Đầu ra mô hình:** xác suất phân phối trên 3 nhãn → chọn nhãn xác suất cao nhất.

### Code thực hiện

**Định nghĩa BiLSTM** — `Tuan1/bai2/SemEval2014_DL.py`, dòng 319–330:
```python
seq_in = Input(shape=(MAX_SEQ_LEN,), name='bilstm_in')
emb    = layers.Embedding(vocab_actual, EMBED_DIM, mask_zero=True)(seq_in)
emb    = layers.SpatialDropout1D(0.3)(emb)
bi1    = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2))(emb)
bi2    = layers.Bidirectional(layers.LSTM(LSTM_UNITS // 2, dropout=0.2))(bi1)
x      = layers.Dense(128, activation='relu')(bi2)
x      = layers.Dropout(0.3)(x)
out    = layers.Dense(n_classes, activation='softmax')(x)
```

**Định nghĩa TextCNN** — dòng 358–373:
```python
for k in CNN_KERNELS:  # [2, 3, 4]
    c = layers.Conv1D(CNN_FILTERS, k, activation='relu', padding='same')(emb2)
    c = layers.GlobalMaxPooling1D()(c)
    conv_pool.append(c)
concat = layers.Concatenate()(conv_pool)
```

**Callbacks chống overfitting** — dòng 257–262:
```python
EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
```

**Hyperparameter:** `BATCH_SIZE=64`, `MAX_EPOCHS=15`, `VOCAB_SIZE=15000`, `MAX_SEQ_LEN=100`, `EMBED_DIM=128`, `LSTM_UNITS=128`, `CNN_FILTERS=64`

### Trình bày với giảng viên

> "Dạ thưa thầy/cô, em xây dựng bài toán ABSA — phân tích cảm xúc theo khía cạnh trên dataset SemEval 2014 Restaurant. Điểm khác biệt so với SA thông thường là cùng một câu có thể có nhiều nhãn cảm xúc khác nhau tùy khía cạnh đang nói đến. Em xây dựng 4 kiến trúc Deep Learning là MLP, BiLSTM, TextCNN và BiLSTM kết hợp Attention, mỗi mô hình được huấn luyện độc lập và đánh giá trên cùng một tập Test để so sánh công bằng."

### Câu hỏi phản biện

**Q1: Tại sao cùng một câu lại có thể cho nhãn khác nhau tùy khía cạnh?**  
Vì cảm xúc phụ thuộc vào đối tượng đang được đánh giá. Câu *"food was great but price was high"* — khi aspect là `food` thì positive, khi aspect là `price` thì negative. Mô hình nhận diện được điều này nhờ đầu vào là `câu [SEP] aspect` thay vì chỉ câu đơn thuần.

**Q2: Tại sao chọn BiLSTM thay vì LSTM một chiều?**  
LSTM một chiều chỉ đọc từ trái sang phải, bỏ lỡ ngữ cảnh phía sau từ hiện tại. BiLSTM đọc cả hai chiều nên hiểu được ngữ cảnh đầy đủ hơn. Ví dụ: từ *"cold"* trong *"the beer was cold"* (tốt) khác với *"the soup was cold"* (xấu) — phụ thuộc vào ngữ cảnh xung quanh.

**Q3: Vì sao dùng EarlyStopping?**  
Để tránh overfitting — mô hình học thuộc dữ liệu Train nhưng không tổng quát hóa được trên Test. `patience=4` có nghĩa nếu val_loss không cải thiện sau 4 epoch liên tiếp thì dừng và khôi phục trọng số tốt nhất.

---

## II. EVALUATION METRIC

### Phương pháp và Mục đích

Dự án tính đầy đủ 5 nhóm độ đo trên **tập Test (537 mẫu, giữ nguyên gốc)**:

| Metric | Công thức tóm tắt | Ý nghĩa |
|---|---|---|
| **Accuracy** | Dự đoán đúng / Tổng số mẫu | Tỷ lệ dự đoán đúng toàn bộ |
| **Precision (Macro)** | TB không trọng số Precision từng lớp | Độ chính xác khi mô hình dự đoán một nhãn |
| **Recall (Macro)** | TB không trọng số Recall từng lớp | Tỷ lệ tìm đúng các mẫu thuộc một nhãn |
| **F1 (Macro)** | 2×P×R/(P+R), TB Macro | Cân bằng Precision và Recall, đối xử bình đẳng các lớp |
| **ROC-AUC (Macro)** | Diện tích dưới đường ROC, OvR Macro | Khả năng phân biệt lớp đúng với lớp sai |

**Tại sao dùng Macro?** Dataset mất cân bằng: `positive` chiếm ~60%, `negative` ~22%, `neutral` ~18%. Macro tính trung bình đều cho cả 3 lớp, không để lớp đa số che khuất lớp thiểu số. Nếu dùng Weighted sẽ thiên vị cho lớp `positive`.

**ROC-AUC đa lớp:** Dùng chiến lược **One-vs-Rest (OvR)** — mỗi lớp lần lượt làm lớp positive. Dự án lấy xác suất Softmax (không dùng nhãn cứng) để tính AUC.

### Code thực hiện

**Hàm `get_metrics`** — `SemEval2014_DL.py`, dòng 229–255:
```python
def get_metrics(name, y_true, y_pred, y_score=None):
    acc    = accuracy_score(y_true, y_pred)
    prec_m = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_m  = recall_score(y_true, y_pred,    average='macro', zero_division=0)
    f1_m   = f1_score(y_true, y_pred,        average='macro', zero_division=0)
    # ROC-AUC: dùng xác suất Softmax, chiến lược OvR
    y_bin = label_binarize(y_true, classes=range(n_classes))
    roc   = roc_auc_score(y_bin, y_score, average='macro', multi_class='ovr')
```

### Kết quả thực nghiệm đã kiểm chứng

Nguồn: `Tuan1/results/bai2/dl_results.csv`

| Mô hình | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) | ROC-AUC |
|---|---|---|---|---|---|
| MLP | 0.6667 | 0.4555 | 0.6767 | 0.4565 | 0.7901 |
| BiLSTM | 0.6592 | 0.6013 | 0.6021 | 0.6201 | 0.8070 |
| **TextCNN** | **0.7076** | **0.6354** | 0.6308 | 0.6451 | **0.8300** |
| BiLSTM+Attention | 0.6499 | 0.5910 | 0.5826 | 0.6128 | 0.8159 |
| DL Ensemble | 0.6946 | 0.6327 | 0.6233 | **0.6507** | 0.8298 |

**Đọc `confusion_matrices_dl.png`:** Ô (hàng i, cột j) = số mẫu có nhãn thật là lớp i nhưng dự đoán là lớp j. Đường chéo chính = dự đoán đúng.

**Đọc `roc_curves_dl.png`:** 3 ô cho 3 lớp (negative/neutral/positive). Đường cong càng gần góc trên trái và AUC càng gần 1.0 thì mô hình phân biệt lớp đó càng tốt.

### Trình bày với giảng viên

> "Em đánh giá trên tập Test 537 mẫu, giữ nguyên không augmentation. Em dùng Macro averaging vì dataset mất cân bằng — Weighted sẽ để lớp positive chiếm 60% che khuất hiệu suất thực trên lớp negative và neutral. ROC-AUC đo bằng chiến lược One-vs-Rest, lấy trực tiếp xác suất Softmax đầu ra."

### Câu hỏi phản biện

**Q1: Tại sao Accuracy cao chưa chắc mô hình tốt?**  
Với dataset mất cân bằng, nếu mô hình luôn đoán `positive` thì Accuracy đạt ~60% mà không học gì. F1 Macro sẽ thấp vì Recall của lớp negative và neutral bằng 0. Vì vậy F1 Macro mới là chỉ số đáng tin cậy hơn.

**Q2: Tập Validation dùng để làm gì, Test để làm gì?**  
Validation dùng để theo dõi `val_loss` trong EarlyStopping — không được dùng để báo cáo kết quả cuối cùng. Test là tập **hoàn toàn không tham gia huấn luyện**, chỉ dùng một lần duy nhất để báo cáo kết quả thực nghiệm khách quan.

---

## III. COMPARE MODELS

### Phương pháp

Dự án so sánh **5 mô hình DL (Bài 2) với 4 mô hình ML Baseline (Bài 1)** trên **cùng tập Test** (537 mẫu, `random_state=42`).

### Kết quả thực nghiệm

Nguồn: `Tuan1/results/bai2/full_comparison.csv`

| Mô hình | Accuracy | F1 (Macro) | ROC-AUC | Loại |
|---|---|---|---|---|
| Logistic Regression | **0.7244** | 0.6035 | **0.8355** | ML Baseline |
| **TextCNN** | 0.7076 | **0.6354** | 0.8300 | Deep Learning |
| Linear SVM | 0.7151 | 0.6154 | 0.8184 | ML Baseline |
| DL Ensemble | 0.6946 | 0.6327 | 0.8298 | Deep Learning |
| Naive Bayes | 0.6983 | 0.5361 | 0.8202 | ML Baseline |
| BiLSTM | 0.6592 | 0.6013 | 0.8070 | Deep Learning |
| BiLSTM+Attention | 0.6499 | 0.5910 | 0.8159 | Deep Learning |
| Random Forest | 0.6797 | 0.5318 | 0.7861 | ML Baseline |
| MLP | 0.6667 | 0.4555 | 0.7901 | Deep Learning |

**Nhận xét trung thực:**
- TextCNN đạt **F1 Macro cao nhất toàn bộ** (0.6354), vượt tất cả ML Baseline.
- Logistic Regression vẫn dẫn đầu về Accuracy và ROC-AUC — tập dữ liệu nhỏ (~2000 gốc) khiến DL chưa vượt trội hoàn toàn.
- DL Ensemble đạt Recall cao nhất (0.6507) — tốt nhất cho bài toán cần bao phủ đều các lớp.

**Biểu đồ:** `results/bai2/model_comparison_dl_vs_ml.png` — 5 cột so sánh, có đường phân cách DL/ML.

### Code thực hiện

**Load kết quả Bài 1 và concat** — `SemEval2014_DL.py`, dòng 501–515:
```python
if os.path.exists(BAI1_CSV):
    df_bai1 = pd.read_csv(BAI1_CSV, index_col='Model')
    df_bai1_cmp['Type'] = 'ML Baseline'
    df_compare = pd.concat([df_compare_dl, df_bai1_cmp])
df_compare.to_csv(os.path.join(RESULTS_DIR, "full_comparison.csv"))
```

### Trình bày với giảng viên

> "Em so sánh 5 mô hình DL với 4 ML Baseline từ Bài 1 trên cùng tập Test, cùng điều kiện chia dữ liệu và random seed 42. TextCNN đạt F1 Macro cao nhất là 0.6354, vượt tất cả mô hình ML. Tuy nhiên về Accuracy, Logistic Regression vẫn cao hơn — đây là giới hạn khi tập dữ liệu còn nhỏ, DL cần nhiều dữ liệu hơn để phát huy hết."

### Câu hỏi phản biện

**Q1: Tại sao Logistic Regression cạnh tranh được với Deep Learning?**  
Dataset gốc chỉ có ~2000 mẫu — quá nhỏ để DL tận dụng hết khả năng học biểu diễn phức tạp. DL cần hàng chục nghìn mẫu để vượt trội hẳn. Trên dữ liệu nhỏ, mô hình tuyến tính với TF-IDF đặc trưng tốt vẫn rất cạnh tranh.

**Q2: So sánh có công bằng không khi feature engineering khác nhau?**  
Cùng Train/Val/Test split và random seed — công bằng về điều kiện đánh giá. Về feature thì khác: MLP dùng TF-IDF+Lexicon, BiLSTM/CNN dùng Word Embedding tự học. Đây là so sánh **pipeline tổng thể** (end-to-end), không phải so sánh thuần kiến trúc.

---

## IV. DATA AUGMENTATION & LÀM SẠCH DỮ LIỆU

### 1. Quy trình Làm sạch Dữ liệu (Data Cleaning)

Trước khi thực hiện làm giàu (augmentation) hay đưa vào mô hình Deep Learning, dữ liệu thô từ SemEval 2014 được xử lý qua pipeline làm sạch toàn diện gồm **6 bước** trong `Tuan1/dataPreprocessing/data_preprocessing.py` (dòng 71–137):

| Bước | Thao tác làm sạch | Mục đích & Chi tiết xử lý | Dòng code |
|---|---|---|---|
| **1. Loại bỏ dữ liệu thiếu (NaN)** | `df.dropna(subset=['Sentence', 'Aspect Term', 'polarity'])` | Xóa các hàng thiếu câu đánh giá, khía cạnh hoặc nhãn cảm xúc | L81 |
| **2. Lọc nhãn nhiễu (`conflict`)** | `df[df['polarity'].isin(['positive', 'negative', 'neutral'])]` | Loại bỏ 91 mẫu mang nhãn `conflict` (chiếm tỉ lệ rất nhỏ, đa nghĩa, gây nhiễu cho bài toán phân loại 3 lớp) | L86 |
| **3. Loại bỏ trùng lặp (Duplicates)** | `df.drop_duplicates(subset=['Sentence', 'Aspect Term', 'polarity'])` | Loại bỏ 25 hàng trùng lặp hoàn toàn để tránh mô hình bị thiên vị (bias) hoặc học vẹt | L91 |
| **4. Chuẩn hóa & Làm sạch văn bản (`clean_text`)** | Regex + `unicodedata.normalize('NFC')` + `.lower()` | Chuẩn hóa bảng mã, xóa HTML, URL, email, ký tự rác và đưa về chữ thường | L95–114 |
| **5. Chuẩn hóa Aspect Term (`clean_aspect`)** | Chuẩn hóa NFC, xóa khoảng trắng thừa, `.lower()` | Giúp từ ngữ chỉ khía cạnh đồng nhất tuyệt đối với từ trong câu | L116–122 |
| **6. Lọc văn bản quá ngắn** | `df[df['Sentence_clean'].str.len() >= 5]` | Loại bỏ các câu bị rỗng hoặc dưới 5 ký tự sau khi làm sạch vì không đủ ngữ cảnh | L132 |

#### Chi tiết kỹ thuật của hàm `clean_text`:
```python
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Chuẩn hóa Unicode về dạng chuẩn NFC (tránh lỗi font, gõ sai chuẩn)
    text = unicodedata.normalize('NFC', text)
    # 2. Xóa các thẻ HTML (ví dụ: <br>, <p>, <div>)
    text = re.sub(r'<[^>]+>', ' ', text)
    # 3. Xóa các mã thực thể HTML (ví dụ: &amp;, &quot;, &gt;)
    text = re.sub(r'&[a-z]+;', ' ', text)
    # 4. Xóa URLs và địa chỉ email
    text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+\.\S+', ' ', text)
    # 5. Xóa các ký tự điều khiển không in được (ASCII control characters)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    # 6. Rút gọn nhiều khoảng trắng thừa, tab, xuống dòng thành 1 dấu cách đơn
    text = re.sub(r'\s+', ' ', text).strip()
    # 7. Chuyển toàn bộ về chữ thường (lowercase) để thống nhất từ vựng
    text = text.lower()
    return text
```

> **Lưu ý đặc biệt quan trọng về Tiền xử lý cho Deep Learning:**  
> Dự án **KHÔNG xóa dấu câu và KHÔNG loại bỏ stopwords** ở bước làm sạch. Trong bài toán ABSA và các mô hình chuỗi (BiLSTM, Attention), các từ như *"not"*, *"never"*, *"but"* và dấu *"!"*, *"?"* mang thông tin ngữ cảnh và cảm xúc cực kỳ quan trọng (ví dụ: *"not good"* trái ngược hoàn toàn với *"good"*; *"great!"* mang sắc thái mạnh hơn *"great"*). Xóa stopwords sẽ phá vỡ cấu trúc ngữ pháp và làm mô hình mất khả năng hiểu ngữ cảnh.

---

### 2. Quy trình Tăng cường Dữ liệu (Data Augmentation)

#### Nguyên tắc chống rò rỉ dữ liệu (Anti-Data Leakage)
Dự án tuân thủ nghiêm ngặt quy trình chuẩn khoa học dữ liệu (dòng 140–163 và 370–382):
1. **Chia dữ liệu TRƯỚC:** Chia toàn bộ dataset sạch (3.577 mẫu) thành 3 tập độc lập:
   - **Train:** 70% (~2.503 mẫu gốc)
   - **Validation:** 15% (537 mẫu)
   - **Test:** 15% (537 mẫu)
2. **Augmentation CHỈ thực hiện trên tập Train:**
   - Mỗi mẫu Train gốc được nhân bản sinh ra thêm 2 mẫu biến thể (`AUG_PER_SAMPLE = 2`)
   - Tập Train sau khi tăng cường đạt **5.985 mẫu** (tăng thêm 3.482 mẫu biến thể), giúp các mô hình DL có đủ dữ liệu để tối ưu hàng chục nghìn trọng số.
3. **Validation và Test GIỮ NGUYÊN GỐC:** Hoàn toàn không qua augmentation để đảm bảo kết quả đánh giá phản ánh chính xác hiệu năng trên dữ liệu thực tế tự nhiên.

#### 5 Kỹ thuật Augmentation được áp dụng
Trong file `Tuan1/dataPreprocessing/data_preprocessing.py`:

| STT | Kỹ thuật | Hàm | Nguyên lý hoạt động | Ví dụ minh họa |
|---|---|---|---|---|
| 1 | **Synonym Replacement** | `synonym_replacement(sentence, n=2)` | Thay 2 từ ngẫu nhiên bằng từ đồng nghĩa từ từ điển ngữ cảnh nhà hàng | *"the **food** was **great**"* → *"the **cuisine** was **excellent**"* |
| 2 | **Random Deletion** | `random_deletion(sentence, p=0.10)` | Mỗi từ có xác suất 10% bị xóa ngẫu nhiên (tăng tính chịu lỗi) | *"service was **very** slow and rude"* → *"service was slow and rude"* |
| 3 | **Random Swap** | `random_swap(sentence, n=1)` | Đổi vị trí 2 từ ngẫu nhiên (chống học vẹt trật tự từ cứng nhắc) | *"the steak was **perfectly cooked**"* → *"the cooked was **perfectly steak**"* |
| 4 | **Pseudo Back-Translation** | `pseudo_back_translation(sentence)` | Paraphrase câu bằng luật biến đổi cấu trúc ngữ nghĩa *(dựa trên luật, không cần mô hình dịch nặng)* | *"**i think** the food was really good"* → *"**i believe** the cuisine seemed quite good"* |
| 5 | **Aspect-Aware Augmentation** | `aspect_aware_augment(sentence, aspect)` | Chèn thêm tiền tố dẫn dắt tập trung vào khía cạnh cần đánh giá | (Aspect: *food*) *"the food was excellent"* → *"**regarding food,** the cuisine was excellent"* |

```python
# Đoạn code thực hiện Augmentation chỉ trên Train (data_preprocessing.py: L370-382)
aug_rows = []
for _, row in df_train_raw.iterrows():
    # Sinh 2 bản biến thể cho mỗi dòng bằng các kỹ thuật ngẫu nhiên
    aug_rows.extend(augment_sample(row, n=AUG_PER_SAMPLE))

df_aug = pd.DataFrame(aug_rows)
# Gộp tập train gốc và tập biến thể
df_train = pd.concat([df_train_raw, df_aug], ignore_index=True)
```

---

### 3. Kết quả và Trực quan hóa

Các biểu đồ được tự động sinh ra và lưu tại `results/`:
- `results/augmentation_comparison.png`: Biểu đồ cột so sánh số lượng mẫu trước và sau augmentation theo 3 nhãn cảm xúc.
- `results/augmentation_techniques.png`: Biểu đồ phân bố đóng góp tần suất áp dụng của 5 kỹ thuật augmentation.
- `results/split_label_distribution.png`: Phân bố tỷ lệ nhãn cân bằng giữa các tập Train, Validation và Test.
- `results/dataset_split_summary.png`: Tóm tắt kích thước từng tập dữ liệu trong toàn bộ pipeline.

---

### 4. Trình bày với giảng viên

> "Dạ thưa thầy/cô, về khâu tiền xử lý và tăng cường dữ liệu, em xây dựng pipeline gồm 2 giai đoạn tách biệt:  
> 1. **Làm sạch dữ liệu:** Em loại bỏ các bản ghi khuyết thiếu (NaN), lọc bỏ các mẫu trùng lặp và loại bỏ 91 mẫu mang nhãn `conflict` gây nhiễu để quy về bài toán 3 lớp chuẩn. Văn bản được chuẩn hóa Unicode NFC, loại bỏ thẻ HTML, URL, email, ký tự điều khiển và chuyển về chữ thường. Em chủ ý **giữ nguyên stopwords và dấu câu** vì trong mô hình Deep Learning chuỗi (BiLSTM, Attention), các từ phủ định như *'not', 'never'* hay liên từ chuyển hướng *'but'* đóng vai trò quyết định đến sắc thái cảm xúc.  
> 2. **Tăng cường dữ liệu (Data Augmentation):** Để tránh rò rỉ dữ liệu (data leakage), em chia tập Train (70%), Val (15%), Test (15%) trước. Sau đó chỉ thực hiện augmentation trên tập Train với 5 kỹ thuật: thay từ đồng nghĩa domain nhà hàng, xóa từ ngẫu nhiên 10%, hoán đổi từ, paraphrase theo luật (pseudo back-translation) và tăng cường có nhận thức khía cạnh (aspect-aware). Mỗi mẫu train sinh thêm 2 biến thể, giúp tập train tăng từ 2.503 lên 5.985 mẫu, tạo đủ khối lượng dữ liệu cho mạng Deep Learning học biểu diễn mà vẫn bảo đảm tập Test hoàn toàn nguyên bản."

---

### 5. Câu hỏi phản biện có thể gặp

**Q1: Tại sao loại bỏ nhãn `conflict` thay vì giữ lại phân loại 4 lớp?**  
Nhãn `conflict` chỉ chiếm một tỷ lệ cực nhỏ (~91 mẫu trong hơn 3.600 mẫu) và thường là các câu vừa khen vừa chê chung chung gây mâu thuẫn lớn cho hàm mất mát. Loại bỏ `conflict` giúp chuẩn hóa bài toán về phân loại 3 lớp rõ ràng (`positive`, `negative`, `neutral`) — đây là chuẩn benchmark phổ biến nhất trong các bài báo khoa học về SemEval 2014 Task 4 ABSA.

**Q2: Tại sao không xóa Stopwords và Dấu câu trong bước làm sạch?**  
Trong Machine Learning truyền thống dùng Bag-of-Words, người ta thường xóa stopwords để giảm số chiều ma trận. Nhưng với Deep Learning xử lý chuỗi (BiLSTM, TextCNN, Attention), cấu trúc ngữ pháp và thứ tự từ mang tính sống còn. Các từ như *"not"*, *"never"*, *"but"*, *"hardly"* đóng vai trò đảo ngược hoặc chuyển hướng cảm xúc. Nếu xóa stopwords, *"not good"* sẽ trở thành *"good"*, khiến mô hình dự đoán sai hoàn toàn. Dấu câu như *"!"* cũng thể hiện cường độ cảm xúc mạnh mẽ.

**Q3: Tại sao phải chia tập Train/Val/Test trước rồi mới thực hiện Data Augmentation?**  
Nếu augment trên toàn bộ dataset trước khi chia, các biến thể sinh ra từ cùng một câu gốc có thể vừa nằm ở tập Train vừa lọt vào tập Test/Val. Điều này gây ra hiện tượng **Data Leakage (Rò rỉ dữ liệu)** nghiêm trọng: mô hình sẽ "học vẹt" câu ở tập Train và đạt điểm cao ảo trên tập Test, nhưng khi áp dụng vào thực tế sẽ thất bại. Do đó, nguyên tắc khoa học bắt buộc là chỉ augment trên tập Train, còn tập Test và Val phải giữ nguyên gốc 100%.

**Q4: Làm sao biết các kỹ thuật Augmentation không làm thay đổi nhãn cảm xúc gốc của câu?**  
Trong code (dòng 357 `data_preprocessing.py`), các mẫu sinh ra được gán mặc định nhãn của câu gốc. Để đảm bảo tính bảo toàn nhãn, các kỹ thuật được thiết kế có kiểm soát:
- *Synonym Replacement:* Chỉ tra cứu trong từ điển đồng nghĩa domain nhà hàng, không dùng từ có sắc thái ngược lại.
- *Random Deletion:* Giới hạn tỷ lệ xóa thấp (chỉ 10%) để giữ nguyên cấu trúc khung của câu.
- *Aspect-Aware:* Chỉ thêm ngữ cảnh tiền tố làm rõ chủ thể, không làm thay đổi vị ngữ đánh giá.  
Tuy nhiên, đây vẫn là phương pháp tăng cường heuristic; trên thực tế vẫn có thể có tỷ lệ nhỏ câu bị biến đổi sắc thái nhẹ — đây là một giới hạn thành thật có thể nêu ra và đề xuất hướng cải thiện bằng LLM paraphrase có kiểm soát nhãn trong tương lai.

**Q5: Pseudo Back-Translation trong code có phải là dịch máy qua lại giữa hai ngôn ngữ không?**  
Không phải. Dịch máy 2 chiều thực thụ (ví dụ Anh → Pháp → Anh) đòi hỏi phải tải các mô hình Translation lớn (như MarianMT, mBART) rất tốn tài nguyên tính toán. Trong code dự án, đây là **Pseudo Back-Translation** (dịch ngược giả lập): sử dụng các bộ quy tắc regex để thay thế cụm từ, chuẩn hóa từ phủ định (ví dụ *"isn't"* → *"is not"*, *"i think"* → *"i believe"*) nhằm tạo ra câu diễn đạt tương đương một cách nhanh chóng và gọn nhẹ.

---

## V. FEATURE ENGINEERING

### Phương pháp — Thực tế trong dự án

> **Lưu ý:** Dự án **KHÔNG dùng GloVe hay BERT pretrained**. Thay vào đó dùng 3 phương pháp:

| Feature | Dùng cho | Kỹ thuật |
|---|---|---|
| **Word Tokenization + Padding** | BiLSTM, TextCNN, BiLSTM+Attention | Keras `Tokenizer` + `pad_sequences` |
| **TF-IDF (Unigram + Bigram)** | MLP | `TfidfVectorizer(max_features=8000, ngram_range=(1,2))` |
| **Sentiment Lexicon (8 chiều)** | MLP | Đếm từ tích cực/tiêu cực/phủ định thủ công |

### Biểu diễn khía cạnh với `[SEP]`

**Code** — `data_preprocessing.py`, dòng 128:
```python
df['text'] = df['Sentence_clean'] + ' [SEP] ' + df['Aspect_clean']
```

**Ví dụ:**
```
Câu gốc: "the food was excellent but service was slow"
Aspect Term: "food"
Chuỗi đầu vào: "the food was excellent but service was slow [SEP] food"
```

**Phân biệt quan trọng:**
- `[SEP]` trong code này là **5 ký tự thông thường** do code ghép vào — Keras Tokenizer xử lý như từ bình thường.
- `[SEP]` của BERT là **special token riêng biệt** có embedding được pretrain, hoàn toàn khác nhau.

**Tại sao cần aspect trong input?** Cùng câu trên nếu aspect là `service` thì nhãn là `negative`. Không có aspect, mô hình không biết đang đánh giá khía cạnh nào.

### Trainable Word Embedding

**Code** — `SemEval2014_DL.py`, dòng 145–153 (Tokenizer) và dòng 320 (Embedding):
```python
# Tokenizer
tokenizer = Tokenizer(num_words=15000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts_train)  # fit CHỈ trên Train
X_train_seq = pad_sequences(..., maxlen=100, padding='post')

# Embedding khởi tạo ngẫu nhiên, tự học trong training
emb = layers.Embedding(vocab_actual, EMBED_DIM=128, mask_zero=True)(seq_in)
```

Đây là **Trainable Embedding từ scratch** — không load weights từ GloVe hay BERT.

### TF-IDF + Lexicon Features (cho MLP)

**Code** — dòng 166–224:
```python
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2), sublinear_tf=True)
# Lexicon: 8 features (n_pos, n_neg, n_negate, pos/n, neg/n, net_score, neg_density, length)
X_train_mlp = np.hstack([X_train_tfidf, X_train_lex])  # 8008 chiều tổng
```

### Trình bày với giảng viên

> "Em dùng 3 phương pháp feature engineering. Cho mô hình sequence như BiLSTM và TextCNN, em dùng Keras Tokenizer chuyển câu thành chuỗi số nguyên, padding về độ dài 100, rồi qua Embedding layer tự học trong quá trình huấn luyện — không dùng GloVe hay BERT pretrained, đây là điểm hạn chế có thể cải thiện. Cho MLP, em kết hợp TF-IDF 8000 chiều với 8 đặc trưng từ điển cảm xúc thủ công."

### Câu hỏi phản biện

**Q1: `[SEP]` trong code này có giống `[SEP]` của BERT không?**  
Không. `[SEP]` trong BERT là special token được tokenizer nhận diện riêng, tạo embedding có ý nghĩa ngữ nghĩa học từ hàng tỷ câu. Trong dự án, `[SEP]` là chuỗi ký tự thông thường do code ghép vào, Keras Tokenizer xử lý như một "từ" bình thường. Tác dụng chỉ là tạo ranh giới phân cách trong văn bản.

**Q2: Tại sao không dùng GloVe hay BERT mà chỉ dùng Trainable Embedding?**  
GloVe và BERT đòi hỏi tải file nặng và cấu hình thêm. Trainable Embedding đơn giản hơn và tự thích nghi với domain nhà hàng. Hạn chế là cần nhiều dữ liệu hơn để học tốt. Dùng GloVe pretrained làm giá trị khởi tạo sẽ giúp hội tụ nhanh hơn — đây là hướng cải thiện tiếp theo.

---

## VI. ENSEMBLE METHODS

### Phân biệt: Kiến trúc kết hợp vs. Ensemble thực sự

| | Kiến trúc kết hợp | Ensemble (dự án này) |
|---|---|---|
| Số computational graph | 1 | 3 riêng biệt |
| Số lần `model.fit()` | 1 | 3 lần độc lập |
| Cách kết hợp | Ghép layer trong cùng mô hình | Trung bình xác suất đầu ra |
| Ví dụ | TextCNN có 3 Conv1D bên trong | BiLSTM + TextCNN + Attn riêng nhau |

### Phương pháp — Soft Voting

3 mô hình BiLSTM, TextCNN, BiLSTM+Attention được huấn luyện **hoàn toàn độc lập**. Khi dự đoán, lấy **trung bình cộng xác suất Softmax** của 3 mô hình.

### Code thực hiện

**3 mô hình fit độc lập** — dòng 332, 375, 441:
```python
bilstm_model = Model(...)
bilstm_model.fit(X_train_seq, ...)   # Lần 1

cnn_model = Model(...)
cnn_model.fit(X_train_seq, ...)      # Lần 2

attn_model = Model(...)
attn_model.fit(X_train_seq, ...)     # Lần 3
```

**Soft Voting** — `SemEval2014_DL.py`, dòng 466–467:
```python
ens_proba = (bilstm_proba + cnn_proba + attn_proba) / 3.0
ens_pred  = np.argmax(ens_proba, axis=1)
```

**Tại sao không có MLP trong Ensemble?** MLP nhận input TF-IDF 8008 chiều — khác với BiLSTM/TextCNN/Attention nhận chuỗi số 100 token. Soft Voting yêu cầu các mô hình cho ra cùng không gian xác suất, nhưng quy trình tiền xử lý đầu vào quá khác nhau.

### Kết quả thực nghiệm

DL Ensemble đạt **Recall Macro cao nhất (0.6507)** và **F1 Macro 0.6327** — tốt về độ bao phủ lớp thiểu số. TextCNN đơn lẻ có F1 cao hơn (0.6354) nhưng Recall thấp hơn (0.6451).

**Biểu đồ:**
- `results/bai2/val_accuracy_dl.png`: Best Validation Accuracy của từng mô hình.
- `results/bai2/training_history.png`: Loss và Accuracy theo epoch của 4 mô hình.

### Trình bày với giảng viên

> "Dự án có Ensemble thực sự — 3 mô hình BiLSTM, TextCNN và BiLSTM+Attention được huấn luyện hoàn toàn độc lập. Sau đó em lấy trung bình cộng xác suất Softmax của 3 mô hình rồi chọn nhãn có xác suất cao nhất — đây là Soft Voting. Mỗi mô hình có thế mạnh riêng nên kết hợp giúp giảm thiểu lỗi mà mô hình đơn lẻ hay mắc."

### Câu hỏi phản biện

**Q1: Có trường hợp nào Ensemble tệ hơn mô hình đơn không?**  
Có. Nếu 3 mô hình đều mắc cùng loại lỗi (correlated errors), trung bình xác suất không giúp ích gì. Kết quả thực tế: TextCNN đơn lẻ có F1 cao hơn Ensemble (0.6354 vs 0.6327) nhưng Ensemble có Recall cao hơn — tùy metric quan tâm mà chọn mô hình.

**Q2: Sự khác nhau giữa Soft Voting và Hard Voting là gì?**  
Hard Voting: mỗi mô hình bỏ phiếu một nhãn cứng, nhãn nào được nhiều phiếu nhất thì chọn. Soft Voting: trung bình xác suất rồi chọn nhãn cao nhất — tốt hơn vì tận dụng được mức độ tự tin của từng mô hình. Ví dụ: BiLSTM rất tự tin (90%) về `positive` sẽ ảnh hưởng nhiều hơn TextCNN chỉ tự tin 55%.

---

## VII. BẢNG TỔNG KẾT

| Yêu cầu | Trạng thái thực tế | Bằng chứng | Việc cần hoàn thiện |
|---|---|---|---|
| **Build Model** | ✅ Đã triển khai và chạy | 4 mô hình DL + 1 Ensemble trong `SemEval2014_DL.py`; kết quả trong `dl_results.csv` | Không bắt buộc |
| **Evaluation Metric** | ✅ Đầy đủ 5 metric, có biểu đồ | Hàm `get_metrics()` tính Accuracy, Precision, Recall, F1, ROC-AUC; `confusion_matrices_dl.png` và `roc_curves_dl.png` đã sinh | Không bắt buộc |
| **Compare Models** | ✅ So sánh DL vs ML Baseline | `full_comparison.csv` có 9 mô hình; `model_comparison_dl_vs_ml.png` đã sinh | Không bắt buộc |
| **Data Augmentation & Làm sạch** | ✅ Đã triển khai và chạy | Pipeline làm sạch 6 bước + 5 kỹ thuật Augmentation trong `data_preprocessing.py`; các biểu đồ phân bố và so sánh đã sinh | Thực nghiệm ablation (có/không aug) để chứng minh hiệu quả |
| **Feature Engineering** | ⚠️ Một phần | TF-IDF, Lexicon, Trainable Embedding đã dùng | **GloVe/BERT chưa triển khai** — cần nói rõ dùng Trainable Embedding từ scratch thay thế |
| **Ensemble Methods** | ✅ Đã triển khai đúng nghĩa | Soft Voting 3 mô hình độc lập tại dòng 466; kết quả trong `dl_results.csv` | Không bắt buộc |

---

## Lời kết luận thuyết trình

> Dự án thực hiện bài toán Aspect-Based Sentiment Analysis trên dataset SemEval 2014 Restaurant, phân loại cảm xúc theo từng khía cạnh được nhắc đến trong câu thay vì gán một nhãn chung cho cả câu. Em đã xây dựng và huấn luyện 4 kiến trúc Deep Learning — MLP, BiLSTM, TextCNN và BiLSTM kết hợp Self-Attention — cùng mô hình Ensemble bằng phương pháp Soft Voting, tất cả đánh giá trên cùng tập Test 537 mẫu độc lập. Kết quả tốt nhất về F1 Macro đạt được bởi TextCNN với 0.6354, vượt trội hơn tất cả mô hình ML truyền thống. Dữ liệu huấn luyện được làm giàu qua 5 kỹ thuật augmentation, áp dụng riêng trên tập Train để đảm bảo đánh giá khách quan và tránh data leakage. Về Feature Engineering, dự án dùng Trainable Word Embedding từ scratch thay vì GloVe hay BERT pretrained — đây là điểm có thể cải thiện tiếp theo để tăng chất lượng biểu diễn từ và có thể nâng cao hiệu suất mô hình trên tập dữ liệu nhỏ. Toàn bộ kết quả, biểu đồ và file so sánh đều được lưu trong thư mục `results/` và có thể tái tạo bằng cách chạy lại hai file Python chính trong dự án.

---

*Báo cáo tạo từ phân tích code thực tế trong repository. Số liệu lấy từ kết quả thực nghiệm đã chạy trong `Tuan1/results/bai2/`. Không có số liệu nào được tự tạo.*
