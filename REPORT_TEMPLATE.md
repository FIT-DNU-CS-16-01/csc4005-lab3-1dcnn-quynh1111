# CSC4005 Lab 3 Report – UrbanSound8K with 1D-CNN

## 1. Thông tin sinh viên

- Họ tên: Đinh Trọng Quỳnh
- Mã sinh viên: 1771040020
- Lớp: KHMT 17-01
- Link GitHub repo:
- Link W&B run/project: (offline, sẽ sync và cập nhật link)

---

## 2. Mục tiêu thí nghiệm

Mô tả ngắn gọn mục tiêu của lab:

- phân loại âm thanh môi trường trên UrbanSound8K,
- sử dụng MFCC/log-mel làm chuỗi đặc trưng theo thời gian,
- xây dựng và huấn luyện 1D-CNN,
- theo dõi thí nghiệm bằng W&B,
- phân tích lỗi bằng confusion matrix.

---

## 3. Dữ liệu và tiền xử lý

### 3.1. Dataset

- Dataset: UrbanSound8K
- Số lớp: 10
- Các lớp: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
- Fold dùng để train: 1–8
- Fold dùng để validation: 9
- Fold dùng để test: 10

### 3.2. Tiền xử lý audio (baseline MFCC)

Điền cấu hình đã dùng:

| Thành phần | Giá trị |
|---|---|
| Sample rate | 16000 |
| Duration | 4.0 |
| Feature type | MFCC |
| n_mfcc / n_mels | 40 / 64 |
| n_fft | 1024 |
| hop_length | 512 |
| Augmentation | bật |

Giải thích ngắn: cần chuẩn hóa sample rate và độ dài để mô hình nhận input có cùng shape, tránh sai lệch phổ và giúp batching ổn định.

Ghi chú cho bài mở rộng:

- Log-mel: giữ nguyên sample rate và duration như baseline, chỉ thay feature_type=logmel, n_mels=64.
- Raw waveform: input là tín hiệu thô, giữ sample rate và duration, batch size nhỏ hơn để chạy ổn trên CPU.

---

## 4. Mô hình 1D-CNN

Mô tả kiến trúc mô hình:

```text
Input feature sequence
→ Conv1D block 1
→ Conv1D block 2
→ Conv1D block 3
→ Global Average Pooling
→ Dense classifier
→ Softmax
```

Bảng cấu hình:

| Thành phần | Giá trị |
|---|---|
| model_name | mfcc_1dcnn |
| hidden_channels | [64, 128, 128] |
| dropout | 0.35 |
| optimizer | adamw |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| batch size | 32 |
| epochs | 12 |
| patience | 4 |

---

## 5. Kết quả thực nghiệm

### 5.1. Kết quả chính

| Metric | Giá trị |
|---|---:|
| Best validation accuracy | 0.5875 |
| Test accuracy | 0.5161 |
| Average epoch time | 15.86 sec |
| Total parameters | 137,930 |
| Trainable parameters | 137,930 |

### 5.2. Learning curves (baseline MFCC)

Chèn hình từ [outputs/mfcc_1dcnn_stable/curves.png](outputs/mfcc_1dcnn_stable/curves.png).

Nhận xét:

- Train loss giảm nhanh, val loss giảm rồi dao động nhẹ.
- Có dấu hiệu overfitting nhẹ (train acc tăng cao trong khi val acc không tăng tương ứng).
- Early stopping xảy ra ở epoch 8.

### 5.3. Confusion matrix (baseline MFCC)

Chèn hình từ [outputs/mfcc_1dcnn_stable/confusion_matrix.png](outputs/mfcc_1dcnn_stable/confusion_matrix.png).

Nhận xét:

- Dễ phân loại: car_horn, gun_shot (recall cao).
- Dễ nhầm: engine_idling hay bị nhầm sang air_conditioner; street_music nhầm sang siren hoặc drilling; children_playing nhầm sang drilling/siren.
- Lý do: các lớp có nền ồn hoặc phổ năng lượng tương tự, cộng với độ dài clip cố định nên phần tín hiệu đặc trưng có thể bị loãng.

### 5.4. Learning curves (log-mel)

Chèn hình từ [outputs/logmel_1dcnn_offline/curves.png](outputs/logmel_1dcnn_offline/curves.png).

Nhận xét:

- Train loss giảm đều, val loss giảm và ổn định hơn baseline.
- Val acc đạt cao hơn baseline (best val acc 0.6263).
- Không có dấu hiệu overfitting mạnh trong các epoch cuối.

### 5.5. Confusion matrix (log-mel)

Chèn hình từ [outputs/logmel_1dcnn_offline/confusion_matrix.png](outputs/logmel_1dcnn_offline/confusion_matrix.png).

Nhận xét:

- Dễ phân loại: gun_shot, jackhammer, siren.
- Dễ nhầm: children_playing nhầm sang street_music; car_horn nhầm sang gun_shot.

### 5.6. Learning curves (raw waveform)

Chèn hình từ [outputs/raw_waveform_1dcnn_offline/curves.png](outputs/raw_waveform_1dcnn_offline/curves.png).

Nhận xét:

- Train loss giảm chậm, val loss dao động, train kéo dài hơn.
- Best val acc thấp hơn baseline (0.4989) và early stopping ở epoch 12.

### 5.7. Confusion matrix (raw waveform)

Chèn hình từ [outputs/raw_waveform_1dcnn_offline/confusion_matrix.png](outputs/raw_waveform_1dcnn_offline/confusion_matrix.png).

Nhận xét:

- Dễ phân loại: engine_idling, air_conditioner.
- Dễ nhầm: dog_bark nhầm sang gun_shot; children_playing nhầm sang street_music.

---

## 6. W&B tracking

Dán link W&B (offline, sẽ sync và bổ sung):

```text
https://wandb.ai/...
```

Ảnh chụp hoặc mô tả dashboard cần có:

- learning curves,
- final metrics,
- configuration,
- confusion matrix image.

---

## 7. Phân tích và thảo luận

Trả lời ngắn các câu hỏi:

1. Vì sao dùng 1D-CNN thay vì MLP cho chuỗi đặc trưng audio?
2. Kernel 1D trong bài này đang trượt theo chiều nào?
3. MFCC giúp mô hình học dễ hơn raw waveform ở điểm nào?
4. Mô hình hiện tại còn hạn chế gì?
5. Có thể cải thiện kết quả bằng cách nào?

Trả lời:

1. 1D-CNN học được mẫu cục bộ theo thời gian (onset, nhịp, phổ thay đổi), MLP không tận dụng tính lân cận theo thời gian.
2. Kernel trượt theo trục thời gian (time_frames).
3. MFCC/log-mel nén phổ và làm trơn, giảm nhiễu và số chiều, giúp mô hình học ổn định hơn raw waveform.
4. Dữ liệu ít và nhiễu, mô hình nhỏ, dễ overfit và nhầm các lớp có phổ tương tự.
5. Tăng dữ liệu/augmentation, thử log-mel, SpecAugment, kiến trúc sâu hơn, tuning LR/regularization.

---

## 8. Bài mở rộng nếu có

Nếu làm raw waveform hoặc log-mel, điền bảng sau:

| Pipeline | Feature/Input | Test accuracy | Nhận xét |
|---|---|---:|---|
| Baseline | MFCC + 1D-CNN | 0.5161 | Val acc tốt nhất 0.5875, có overfitting nhẹ |
| Extension 1 | log-mel + 1D-CNN | 0.6387 | Cải thiện rõ rệt so với MFCC |
| Extension 2 | raw waveform + 1D-CNN | 0.5871 | Train chậm hơn, val acc thấp hơn MFCC, test cao hơn MFCC |

Nhận xét chi tiết cho log-mel:

- Dễ phân loại: gun_shot (recall 1.0), jackhammer (recall 0.72), siren (recall 0.72).
- Dễ nhầm: car_horn hay nhầm sang gun_shot; children_playing hay nhầm sang street_music và drilling; street_music nhầm sang children_playing và siren.
- Xu hướng: log-mel cải thiện độ ổn định và độ chính xác tổng thể so với MFCC.

Nhận xét chi tiết cho raw waveform:

- Dễ phân loại: engine_idling, air_conditioner (recall cao).
- Dễ nhầm: dog_bark nhầm sang gun_shot; children_playing nhầm sang street_music; drilling nhầm sang children_playing/street_music; siren nhầm sang children_playing.
- Xu hướng: train chậm hơn, val acc thấp hơn MFCC nhưng test acc cao hơn; cần nhiều dữ liệu hoặc kiến trúc mạnh hơn để ổn định.

---

## 9. Kết luận

Tóm tắt 3–5 ý chính học được từ lab.

- Pipeline MFCC/log-mel + 1D-CNN chạy ổn trên CPU và cho kết quả hợp lý.
- Log-mel trong run này tốt hơn MFCC về test accuracy.
- Confusion matrix giúp thấy các lớp có phổ giống nhau dễ nhầm.
- Chuẩn hóa sample rate và duration là bắt buộc để batch ổn định.
- W&B giúp theo dõi learning curves và so sánh các cấu hình.
