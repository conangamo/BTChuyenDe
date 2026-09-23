# 📘 CẨM NANG HỌC TẬP VÀ BÁO CÁO: SENTIMENT ANALYSIS (PHÂN TÍCH CẢM XÚC) BẰNG MACHINE LEARNING

> **Mục tiêu tài liệu**: Giúp bạn hiểu tận gốc bản chất đề bài, quy trình làm việc chuẩn trong Xử lý Ngôn ngữ Tự nhiên (NLP) & Học máy (ML), ý nghĩa sâu xa của từng chỉ số đánh giá, lý do cần mô hình cơ sở (baseline), và cách trình bày, biện luận kết quả để thuyết phục giảng viên.

---

## MỤC LỤC
1. [Bản chất đề bài yêu cầu những gì?](#1-bản-chất-đề-bài-yêu-cầu-những-gì)
2. [Cái "bẫy" học thuật: Linear Regression hay Logistic Regression?](#2-cái-bẫy-học-thuật-linear-regression-hay-logistic-regression)
3. [Quy trình thực hiện chuẩn (Machine Learning Pipeline)](#3-quy-trình-thực-hiện-chuẩn-machine-learning-pipeline)
4. [Hiểu tường tận 5 nhóm Evaluation Metrics (Độ đo đánh giá)](#4-hiểu-tường-tận-5-nhóm-evaluation-metrics-độ-đo-đánh-giá)
5. [Mô hình Baseline là gì và tại sao phải so sánh?](#5-mô-hình-baseline-là-gì-và-tại-sao-phải-so-sánh)
6. [Giảng viên thực sự muốn bạn hiểu điều gì ở bài tập này?](#6-giảng-viên-thực-sự-muốn-bạn-hiểu-điều-gì-ở-bài-tập-này)
7. [Hướng dẫn cách trình bày, giải thích và trả lời vấn đáp](#7-hướng-dẫn-cách-trình-bày-giải-thích-và-trả-lời-vấn-đáp)

---

## 1. Bản chất đề bài yêu cầu những gì?

Đề bài:
> *"Build mô hình SA sử dụng ML: Naive Bayes, Linear Regression.*  
> *Evaluation Metric: Đánh giá mô hình bằng nhóm các độ đo trong các phương pháp đánh giá phổ biến: F1, Accuracy, ROC, Precision, Recall.*  
> *Compare models: Tiến hành so sánh với các mô hình baseline trước đây trên cùng dataset."*

Đề bài thực chất giao cho bạn một bài toán kinh điển trong Trí tuệ Nhân tạo: **Xây dựng hệ thống học máy hoàn chỉnh (End-to-End Machine Learning Pipeline) cho bài toán Phân loại Văn bản (Text Classification)**.

Cụ thể chia làm 3 trụ cột:
1. **Dựng pipeline ML**: Dữ liệu chữ (text) $\to$ Vector số hóa $\to$ Huấn luyện thuật toán ML cổ điển.
2. **Đánh giá đa chiều (Multi-metric Evaluation)**: Không đánh giá qua loa bằng mỗi `Accuracy`, mà phải soi xét từng ngóc ngách qua `Precision`, `Recall`, `F1-Score`, và `ROC-AUC`.
3. **Thực nghiệm đối sánh (Benchmarking & Comparison)**: So sánh khách quan giữa các thuật toán trên cùng một tập dữ liệu thử nghiệm (Test set) cố định.

---

## 2. Cái "bẫy" học thuật: Linear Regression hay Logistic Regression?

Trong đề bài có ghi: **"Linear Regression"**. Đây là một điểm **cực kỳ quan trọng** bạn cần hiểu rõ bản chất:

### 2.1. Phân biệt rõ ràng
- **Linear Regression (Hồi quy tuyến tính)**:
  - Dự đoán đầu ra là **số thực liên tục** (Continuous value), ví dụ: giá nhà, dự báo nhiệt độ, doanh thu ($-\infty \to +\infty$).
  - Không sinh ra xác suất phân loại nhãn.
- **Logistic Regression (Hồi quy Logistic)**:
  - Dù mang tên "Regression", đây lại là thuật toán **Phân loại (Classification)**!
  - Nó sử dụng hàm Sigmoid/Softmax để biến đổi giá trị tuyến tính thành **xác suất nằm trong đoạn $[0, 1]$** nhằm phân loại nhãn (`positive`, `negative`, `neutral`, v.v.).

### 2.2. Trong bài tập Sentiment Analysis:
- Bài toán SA trên tập SemEval có 4 nhãn phân loại: `positive`, `negative`, `neutral`, `conflict`. Đây là bài toán **Phân loại đa lớp (Multi-class Classification)**.
- Do đó, thuật toán đúng về mặt toán học và bản chất kỹ thuật phải là **Logistic Regression**.
- *Nếu đề bài ghi "Linear Regression"*: Giảng viên có thể ghi nhầm theo thói quen gọi chung các mô hình tuyến tính (Linear Models), hoặc cố tình để kiểm tra xem sinh viên có phân biệt được hay nhắm mắt làm bừa.
- **Cách trả lời giảng viên**: *"Dạ thưa thầy/cô, vì SA là bài toán phân loại nhãn rời rạc chứ không phải dự đoán giá trị liên tục, nên em đã áp dụng Logistic Regression (thuộc họ Linear Models) để dự đoán xác suất và phân lớp."*

---

## 3. Quy trình thực hiện chuẩn (Machine Learning Pipeline)

Một bài làm đạt chuẩn không phải chỉ viết vài dòng thư viện rồi in kết quả, mà cần tuân thủ 6 bước logic:

```
[1. Khám phá & Tiền xử lý]
         │
         ▼
[2. Tách Train/Test có Stratify]
         │
         ▼
[3. Trích xuất đặc trưng (TF-IDF)]
         │
         ▼
[4. Huấn luyện các mô hình (NB, LR, SVM, RF)]
         │
         ▼
[5. Đánh giá đa độ đo (Acc, P, R, F1, ROC)]
         │
         ▼
[6. Trực quan hóa & Phân tích lỗi (Error Analysis)]
```

### Chi tiết từng bước:
1. **Khám phá dữ liệu (EDA) & Tiền xử lý (Preprocessing)**:
   - Đọc dữ liệu, kiểm tra số lượng mẫu mỗi nhãn.
   - Làm sạch văn bản: chuyển chữ thường (lowercase), loại bỏ ký tự lạ, kết hợp câu gốc (`Sentence`) cùng khía cạnh đánh giá (`Aspect Term`) để mô hình hiểu đúng ngữ cảnh.
2. **Chia tập Train/Test (Train-Test Split)**:
   - Chia tỷ lệ thông dụng: $80\%$ Train - $20\%$ Test.
   - **Bắt buộc dùng `stratify=y`**: Vì nhãn `conflict` rất ít ($2.5\%$), nếu chia ngẫu nhiên bình thường, tập Test có thể hoàn toàn không có mẫu nào của nhãn này!
3. **Vector hóa văn bản (Feature Extraction - TF-IDF)**:
   - Máy tính không đọc được chữ, chỉ đọc được số.
   - **TF-IDF (Term Frequency - Inverse Document Frequency)** gán trọng số cho từng từ: từ nào xuất hiện nhiều trong 1 câu nhưng hiếm trong toàn bộ văn bản thì được điểm cao (chứa nhiều thông tin cảm xúc như: *excellent, terrible, delicious*).
   - Thiết lập `ngram_range=(1, 2)` để bắt được cụm 2 từ (như *"not good"* khác với *"good"*).
   - **Quy tắc vàng chống rò rỉ dữ liệu (Data Leakage)**: Chỉ dùng `.fit_transform()` trên tập Train, còn tập Test chỉ dùng `.transform()`.
4. **Huấn luyện mô hình**:
   - Mô hình yêu cầu: `Multinomial Naive Bayes`, `Logistic Regression`.
   - Mô hình so sánh (Baseline): `Linear SVM`, `Random Forest`.
5. **Đánh giá & Trực quan hóa**:
   - Xuất bảng số liệu tổng hợp `model_results.csv`.
   - Vẽ biểu đồ cột so sánh, biểu đồ Ma trận nhầm lẫn (Confusion Matrix), và đường cong ROC (ROC Curve).

---

## 4. Hiểu tường tận 5 nhóm Evaluation Metrics (Độ đo đánh giá)

Rất nhiều sinh viên chỉ biết chạy hàm ra số mà không hiểu số đó nói lên điều gì. Giảng viên chấm điểm cao nhất ở phần bạn hiểu bản chất các độ đo này:

### 4.1. Nền tảng: Ma trận nhầm lẫn (Confusion Matrix)
Với mỗi nhãn (ví dụ nhãn `positive`):
- **TP (True Positive)**: Thực tế là Positive, mô hình đoán đúng là Positive.
- **TN (True Negative)**: Thực tế KHÔNG phải Positive, mô hình đoán đúng là KHÔNG phải.
- **FP (False Positive - Báo động giả)**: Thực tế không phải, nhưng mô hình đoán nhầm là Positive.
- **FN (False Negative - Bỏ sót)**: Thực tế là Positive, nhưng mô hình lại đoán thành nhãn khác.

---

### 4.2. Accuracy (Độ chính xác tổng thể)
$$\text{Accuracy} = \frac{\text{Số dự đoán đúng toàn bộ}}{\text{Tổng số mẫu}}$$
- **Ý nghĩa**: Tỷ lệ phần trăm đoán trúng trên toàn bộ tập dữ liệu.
- **Điểm yếu chí mạng**: Nếu dữ liệu bị lệch (Imbalanced), ví dụ $80\%$ mẫu là `positive`, mô hình chỉ cần "đoán bừa" toàn bộ là `positive` thì Accuracy vẫn đạt $80\%$, trong khi nó không học được gì về các nhãn còn lại! Do đó, **không bao giờ được dùng mỗi Accuracy để kết luận mô hình giỏi hay dở**.

---

### 4.3. Precision (Độ chuẩn xác / Độ tin cậy)
$$\text{Precision} = \frac{TP}{TP + FP}$$
- **Ý nghĩa trả lời câu hỏi**: *"Trong tất cả những câu mà mô hình nói là Tích cực (Positive), thì thực tế có bao nhiêu % câu thực sự đúng là Tích cực?"*
- **Khi nào cần Precision cao?** Khi cái giá của việc "đoán nhầm/vu khống" (FP) là rất đắt. Ví dụ: Bộ lọc thư rác (Spam filter) không được phép gắn nhầm email quan trọng của khách hàng thành Spam.

---

### 4.4. Recall (Độ bao phủ / Độ nhạy)
$$\text{Recall} = \frac{TP}{TP + FN}$$
- **Ý nghĩa trả lời câu hỏi**: *"Trong tất cả các câu thực tế vốn là Tích cực (Positive), mô hình đã 'bắt' được bao nhiêu % hay là bỏ sót?"*
- **Khi nào cần Recall cao?** Khi cái giá của việc "bỏ sót" (FN) là nguy hiểm. Ví dụ: Chẩn đoán phát hiện ung thư hoặc phát hiện lỗi nghiêm trọng của máy móc (thà bắt nhầm còn hơn bỏ sót).

---

### 4.5. F1-Score (Điểm điều hòa cân bằng)
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Tại sao cần F1-Score?** Vì Precision và Recall luôn có xu hướng đối nghịch nhau (đánh đổi - Trade-off). Muốn tăng Precision thì mô hình phải rất khắt khe, dẫn đến bỏ sót nhiều (Recall giảm). F1-Score là trung bình điều hòa, giúp đo lường mức độ cân bằng giữa cả hai.
- **Macro F1 vs Weighted F1**:
  - `Macro F1`: Tính F1 riêng cho từng nhãn rồi cộng lại chia đều. **Cực kỳ công bằng cho các lớp thiểu số** (như `conflict`, `neutral`). Nếu mô hình bỏ bê lớp nhỏ, Macro F1 sẽ rớt thảm hại ngay lập tức.
  - `Weighted F1`: Tính F1 có nhân theo trọng số số lượng mẫu của từng nhãn.
  - 👉 **Giảng viên luôn nhìn vào `Macro F1` để đánh giá năng lực thực sự của mô hình NLP**.

---

### 4.6. ROC và ROC-AUC (Đặc trưng hoạt động của bộ thu & Diện tích dưới đường cong)
- **Đường cong ROC (Receiver Operating Characteristic)**: Là đồ thị biểu diễn mối tương quan giữa **Tỷ lệ dự đoán đúng lớp dương (True Positive Rate - Recall)** và **Tỷ lệ báo động giả (False Positive Rate)** khi ta dịch chuyển ngưỡng xác suất quyết định (Threshold) từ $0 \to 1$.
- **AUC (Area Under the Curve)**: Diện tích nằm dưới đường cong ROC.
  - $\text{AUC} = 0.5$: Mô hình hoàn toàn vô dụng (ngang với việc tung đồng xu may rủi).
  - $\text{AUC} = 1.0$: Mô hình hoàn hảo tuyệt đối.
  - $\text{AUC} \in [0.8, 0.9]$: Mô hình phân biệt rất tốt giữa các lớp cảm xúc.
- **Ưu điểm lớn nhất của ROC-AUC**: Nó đánh giá chất lượng **xác suất (Confidence score)** mà mô hình trả về, độc lập với việc bạn chọn cắt ngưỡng ở $0.5$ hay $0.7$.

---

## 5. Mô hình Baseline là gì và tại sao phải so sánh?

Trong nghiên cứu khoa học và phát triển sản phẩm công nghệ:
> **"Một con số đứng một mình là một con số vô nghĩa."**

- Nếu bạn nói: *"Mô hình Naive Bayes của em đạt Accuracy 65%"*, người nghe sẽ hỏi: *"65% là cao hay thấp? Có đáng dùng không?"*
- Đó là lý do ta cần **Mô hình Baseline (Đường cơ sở)**:
  1. **Tạo mốc tham chiếu chuẩn**: Để biết mô hình đề xuất có thực sự tốt hơn các thuật toán thông dụng khác hay không.
  2. **Đánh giá trên cùng một điều kiện (Fair Benchmark)**: Cùng tập dữ liệu, cùng cách chia Train/Test, cùng tập từ vựng TF-IDF.
- Trong bài làm này, ta đã chọn:
  - **Baseline 1: Linear SVM**: Thuật toán phân loại văn bản cực kỳ mạnh mẽ trong học máy cổ điển (tìm siêu phẳng phân tách tối ưu).
  - **Baseline 2: Random Forest**: Đại diện cho họ mô hình Ensemble (tập hợp các cây quyết định), giúp kiểm tra xem mô hình dạng cây phi tuyến tính có vượt qua được các mô hình tuyến tính trên văn bản hay không.

---

## 6. Giảng viên thực sự muốn bạn hiểu điều gì ở bài tập này?

Bảng đối chiếu giữa **cách nghĩ sơ sài** và **suy nghĩ của sinh viên hiểu bài**:

| Yếu tố | Sinh viên học vẹt nghĩ gì | Giảng viên muốn sinh viên thực sự hiểu |
|---|---|---|
| **Dữ liệu** | Cứ ném dữ liệu vào model là chạy. | Dữ liệu thực tế luôn **mất cân bằng (Imbalanced)** (`positive` áp đảo, `conflict` chỉ 2.5%). Phải dùng `stratified sampling` và tiền xử lý cẩn thận. |
| **Độ đo** | Thấy Accuracy cao (70%) là khen mô hình tốt. | Nhìn vào **Macro F1 (khoảng 0.32 - 0.43)** để thấy mô hình vẫn gặp rất nhiều khó khăn ở các lớp thiểu số. Hiểu được mặt hạn chế của Accuracy. |
| **Thuật toán** | Code chạy không lỗi là xong. | Hiểu tại sao **Logistic Regression** và **Linear SVM** lại thắng Naive Bayes và Random Forest trên đặc trưng TF-IDF (do dữ liệu văn bản là thưa - sparse và có số chiều rất lớn). |
| **Thực nghiệm** | Chỉ cần ra 1 file kết quả. | Phải có đầy đủ **Artifacts trực quan**: Bảng số liệu CSV, Biểu đồ thanh so sánh, Confusion Matrix chỉ rõ đoán nhầm ở đâu, ROC Curves thể hiện năng lực phân biệt. |

---

## 7. Hướng dẫn cách trình bày, giải thích và trả lời vấn đáp

Nếu phải báo cáo đồ án, thuyết trình hoặc trả lời câu hỏi của giảng viên, bạn hãy dùng bộ khung câu trả lời mẫu dưới đây:

### 7.1. Trình bày tổng quan kết quả thực nghiệm
> *"Kính thưa thầy/cô, trên tập dữ liệu SemEval 2014 Restaurant, nhóm đã triển khai hai mô hình trọng tâm là **Multinomial Naive Bayes** và **Logistic Regression**, đồng thời đối sánh với hai mô hình baseline là **Linear SVM** và **Random Forest**.  
> Kết quả cho thấy:
> - **Logistic Regression** vượt trội hơn hẳn **Naive Bayes** trên toàn bộ các chỉ số: Accuracy tăng từ 64.8% lên 70.5%, Macro F1 tăng mạnh từ 0.3237 lên 0.4315, và ROC-AUC đạt mức rất ấn tượng là **0.8730**.
> - Khi đối sánh với các baseline, **Linear SVM** cho độ chính xác và Macro F1 cao nhất (Acc: 71.5%, F1: 0.5181), trong khi **Logistic Regression** lại có năng lực xếp hạng xác suất tốt nhất với ROC-AUC vươn lên dẫn đầu (0.8730)."*

### 7.2. Giải thích nguyên nhân (Root-cause analysis)
> - **Tại sao Naive Bayes lại có kết quả thấp nhất?**
>   *"Vì Naive Bayes dựa trên giả định 'ngây thơ' rằng các từ xuất hiện độc lập với nhau. Trong văn bản thực tế, các từ luôn đi cùng nhau và bổ trợ ngữ nghĩa (như 'not good', 'hardly recommend'), việc bỏ qua liên kết từ khiến Naive Bayes bị suy giảm độ chính xác."*
> - **Tại sao Random Forest không phải là lựa chọn tối ưu cho bài này?**
>   *"Đặc trưng TF-IDF tạo ra không gian vector thưa (sparse) với 10,000 chiều. Các mô hình tuyến tính như Logistic Regression hay Linear SVM tìm ranh giới phân tách trong không gian nhiều chiều tốt hơn nhiều so với việc phân nhánh theo từng trục của cây quyết định trong Random Forest."*

### 7.3. Phân tích điểm yếu và hướng phát triển (Critical Thinking - Điểm cộng lớn)
> *"Dựa vào Confusion Matrix, nhóm nhận thấy cả 4 mô hình đều dự đoán kém ở nhãn `conflict` (F1 gần bằng 0) và nhãn `neutral`. Nguyên nhân trực tiếp là do hiện tượng mất cân bằng nhãn nghiêm trọng (lớp positive chiếm gần 60%, trong khi conflict chỉ chiếm 2.5%).  
> Để khắc phục trong các giai đoạn sau, có thể áp dụng:
> 1. Kỹ thuật cân bằng dữ liệu: Dùng `class_weight='balanced'` hoặc kỹ thuật sinh mẫu SMOTE.
> 2. Chuyển sang các mô hình ngôn ngữ sâu (Deep Learning / Transformers như BERT, RoBERTa) để nắm bắt ngữ cảnh tốt hơn thay vì chỉ dùng túi từ (Bag-of-Words/TF-IDF)."*

---

## 8. Tóm tắt danh mục tài nguyên hiện có trong thư mục

Dự án của bạn tại `f:\laptrinhPython\baiTapChuyenDe2\Tuan1` hiện đã hoàn thiện đầy đủ 100%:

1. 📄 **Mã nguồn chạy hoàn chỉnh**: [`SemEval2014Restaurant.py`](file:///f:/laptrinhPython/baiTapChuyenDe2/Tuan1/SemEval2014Restaurant.py)
   - Tự động nạp dữ liệu, tiền xử lý, trích xuất TF-IDF, huấn luyện 4 mô hình, tính toán đầy đủ 5 độ đo và lưu kết quả.
2. 📊 **Số liệu thực nghiệm**: [`results/model_results.csv`](file:///f:/laptrinhPython/baiTapChuyenDe2/Tuan1/results/model_results.csv)
   - Lưu trữ điểm số chi tiết của cả 4 mô hình trên tất cả các tiêu chí.
3. 📈 **Hệ thống biểu đồ minh chứng trực quan**:
   - [`results/model_comparison.png`](file:///f:/laptrinhPython/baiTapChuyenDe2/Tuan1/results/model_comparison.png): Biểu đồ so sánh trực quan giữa các mô hình.
   - [`results/confusion_matrices.png`](file:///f:/laptrinhPython/baiTapChuyenDe2/Tuan1/results/confusion_matrices.png): Ma trận nhầm lẫn chi tiết của từng thuật toán.
   - [`results/roc_curves.png`](file:///f:/laptrinhPython/baiTapChuyenDe2/Tuan1/results/roc_curves.png): Đường cong ROC cho từng lớp nhãn cảm xúc.
   - [`results/label_distribution.png`](file:///f:/laptrinhPython/baiTapChuyenDe2/Tuan1/results/label_distribution.png): Thống kê phân bố nhãn Train vs Test.
4. 📘 **Tài liệu học tập & hướng dẫn này**: [`HUONG_DAN_HOC_TAP_SA.md`](file:///f:/laptrinhPython/baiTapChuyenDe2/Tuan1/HUONG_DAN_HOC_TAP_SA.md).
