# T-024 — Kế hoạch benchmark 3 model phát hiện watermark (Codex autonomous smoke test)

**Ngày lập:** 18/10/2026  
**Phạm vi:** phát hiện *watermark nhìn thấy* trên ảnh, đặc biệt chữ `sample`, `preview`, `watermark`, watermark mờ / chéo / góc ảnh; **không** nghiên cứu xóa watermark hoặc phát hiện watermark ẩn (SynthID, WAM).  
**Mục tiêu:** Codex tự triển khai từ repository hiện tại, tự tải **dữ liệu ảnh tổng ~15–25 MB**, chạy **ba pretrained checkpoint chuyên watermark** (không train), xuất kết quả đo đạc tái lập được. Dung lượng checkpoint + môi trường **không** nằm trong hạn mức dữ liệu ảnh.

## 1. Chọn ba model có checkpoint công khai

| ID nội bộ | Model | Dạng | Checkpoint | Nguồn | Ghi chú |
|---|---|---|---:|---|---|
| `yolov8n` | qfisch/yolov8n-watermark-detection | Detection (bbox) | 6.23 MB | https://huggingface.co/qfisch/yolov8n-watermark-detection | YOLOv8 Nano fine-tuned cho watermark chữ và logo; license AGPL-3.0, tác giả nói academic only |
| `efficientnetv2_b0` | qwertyforce/watermark_detection | Binary classification | 23.9 MB | https://huggingface.co/qwertyforce/watermark_detection | EfficientNetV2-B0; model card ghi “use sigmoid”; license CC BY-SA 4.0; checkpoint pickle nguyên model, khả năng kẹt phiên bản `timm` |
| `convnext_tiny` | boomb0om/watermark-detectors, code boomb0om/watermark-detection | Binary classification | 113 MB | https://huggingface.co/boomb0om/watermark-detectors ; https://github.com/boomb0om/watermark-detection | State dict PyTorch, đầu ra 2 logits; license BSD-3-Clause trên HF repo. Báo cáo tác giả: 93.44% trên bộ 122 ảnh — không được coi là so sánh độc lập |

**Lưu ý:** checkpoint YOLO `best.pt` và EfficientNet `model.pt` dùng pickle: có rủi ro khi deserialize, khác với safetensors. Chỉ lấy đúng repo/file đã liệt kê, ghi lại revision + SHA-256; chạy trong venv/conda hoặc container cô lập trên máy được phép, không tự tải/chạy script ngoài danh sách. Nếu phải dùng `torch.load(weights_only=False)` cho EfficientNet, hãy xác nhận checkpoint đúng nguồn và ghi rõ lựa chọn bảo mật ở README; nếu thiết lập bảo mật không cho phép thì đánh dấu *blocked* (không giả vờ hoàn thành). Không chạy untrusted code bằng `trust_remote_code=True` hoặc các lệnh install nguồn không kiểm soát.

## 2. Dataset nhỏ (bắt buộc cùng ảnh cho cả ba)

**Nguồn:** https://huggingface.co/datasets/boomb0om/watermarks-validation

- Repo tổng **12.6 MB**, 122 ảnh: **62 watermark / 60 clean**. Đọc `validation_data.csv` gồm `path,watermark`. File ảnh nằm trong `clean/`, `watermark/`.
- Dùng `huggingface_hub.snapshot_download(repo_id="boomb0om/watermarks-validation", repo_type="dataset", allow_patterns=["clean/**", "watermark/**", "validation_data.csv"], local_dir=...)`. Không tải bộ ~1–2.6 GB như `qwertyforce/scenery_watermarks` hay `Watermark-or-Not-20K`.
- Xác thực đủ **122 ảnh**, đường dẫn từ CSV tồn tại, đọc được bằng PIL, nhãn True/False chuẩn; ghi SHA-256 hoặc manifest; không âm thầm bỏ ảnh bị lỗi. Nếu fail, báo lỗi và vẫn chạy phần còn lại có thể thực thi.
- **Synthetic stress test**: chọn khoảng 20 ảnh clean làm nền (chỉ dùng ảnh bản địa của dataset vừa tải), thêm watermark chữ bằng PIL lên ảnh để tạo **40 positive**: SAMPLE/PREVIEW/WATERMARK, chữ mờ, góc/giữa/chéo, kích thước nhỏ/lớn, opacity khác nhau; thêm **20 negative** giữ nguyên ảnh clean và/hoặc các overlay *không mang nghĩa watermark* (cần review để tránh nhãn nhập nhằng). Số lượng ảnh gốc không tăng do duplicate JPEG chất lượng ~75–85, đảm bảo dữ liệu ảnh **<=25 MB** hoặc bỏ synthetic augmentation khi vượt mức. Seed=42. Sinh `synthetic_manifest.csv` và contact sheet 40–60 ảnh cho người review.
- Tách riêng `public_122` và `synthetic`; không gộp báo cáo làm một accuracy nếu không ghi rõ nguồn. Public set vốn là tập đánh giá tác giả ConvNeXt đã dùng, vì vậy có thể thiên vị model này và không được xem là đo generalization độc lập.

## 3. Chuẩn hóa adapter và prediction

Tạo một adapter cho mỗi model cùng interface:

```python
class WatermarkAdapter:
    name: str
    def load(self, device: str): ...
    def predict(self, image_rgb) -> dict:
        # returns {"score": float in [0, 1], "label": int in {0,1},
        #          "boxes": list, "timing_ms": float}
        ...
```

- `yolov8n`: `hf_hub_download(..., filename="best.pt")` rồi `ultralytics.YOLO(local_file)`. Kiểm tra `model.names` trước khi chạy; chỉ chọn các class watermark được checkpoint cung cấp. Dùng `conf=0.001` để tránh cắt mất các box trước khi áp threshold; img size 640, cùng chính sách ảnh RGB; score cấp ảnh = **max confidence của box watermark**, không có box -> score 0.0. `label = int(score >= 0.5)`. Lưu bbox và confidence để có thể xem lại. Dùng threshold 0.5 làm baseline, đánh giá nhạy cảm threshold 0.25/0.75 nếu muốn nhưng KHÔNG chọn threshold tối ưu bằng chính test set rồi báo điểm tối ưu như khách quan.
- `efficientnetv2_b0`: tải đúng `model.pt`, đọc kiến trúc, lớp cuối và số lượng output. Có thể cần bản `timm` tương thích với class import path trong checkpoint; đừng tự gọi `torch.hub.load` hoặc bỏ qua lỗi. Model card nói `sigmoid`; nếu output logits một chiều thì `sigmoid(logit)` (nếu model đã có sigmoid ở head thì đừng sigmoid thêm); 2 logits thì dùng softmax theo mapping phải được xác thực bằng nhãn gốc và vài ví dụ. Ghi cách preprocess thực tế: kích thước, normalization, RGB. Nếu vẫn không thể suy ra đúng, để trạng thái *blocked_mapping* thay vì bịa kết quả.
- `convnext_tiny`: dựa trên code `wmdetection.models.get_convnext_model` và `convnext-tiny_watermarks_detector.pth`, sử dụng `torch.load(..., map_location='cpu', weights_only=True)` khi checkpoint tương thích; `load_state_dict`. Preprocess từ repo là `Resize((256,256))`, `ToTensor()`, normalize ImageNet `(0.485,0.456,0.406)/(0.229,0.224,0.225)`; đầu ra `softmax(logits)[1]`, vì predictor dự án dùng class index 1 => watermark. `fp16=False`.
- Mọi adapter: PIL RGB; áp dụng `ImageOps.exif_transpose`; `torch.inference_mode()`, `.eval()`; đảm bảo score finite và trong [0,1]. Nếu GPU không khả dụng, chạy CPU và ghi chính xác thiết bị. Không chạy train/fine-tune, không tự đổi pretrained weights sang ImageNet.

## 4. Đo đạc công bằng

- Dùng **cùng 122 ảnh** và cùng synthetic set với ba model; batch size = 1 để mô phỏng lọc ảnh online. Ghi CPU/GPU, PyTorch, package versions, resolution, số thread, model bytes và hash, dataset revision.
- Cấp ảnh: Accuracy, Precision, Recall, F1 cho class watermark=1, FPR (clean bị loại nhầm), TN/FP/FN/TP, confusion matrix. Vì tác vụ **lọc bỏ ảnh chứa watermark**, chú trọng Recall của watermark và FPR của ảnh clean.
- Latency: warmup 5–10 lượt/model, benchmark ít nhất 30 ảnh hoặc toàn bộ 122 ảnh; `time.perf_counter()`, với CUDA phải `torch.cuda.synchronize()` trước/sau. Report `mean`, `median (p50)`, `p95`, throughput ảnh/s; ghi rõ **end-to-end** (preprocess+model+postprocess) và **model-only** khi đo được — đừng gọi nhầm tốc độ nội bộ YOLO thành end-to-end. Không tính thời gian tải checkpoint/warmup vào inference latency; download và load time đo riêng.
- Báo cáo **vấn đề tương thích** và số ảnh chạy thành công của từng adapter. Không tự tạo số liệu nếu model không chạy.
- **Optional:** majority vote nhãn ba model (`>=2` báo watermark), chỉ thực hiện khi đủ 3 model có score hợp lệ trên cùng ảnh; latency sequential ~ tổng latency 3 model. Báo cáo voting so với model đơn và phân tích xem có đáng đổi tốc độ lấy chất lượng không. Đừng dùng voting nếu chưa có 3 model chạy thành công.

## 5. Cấu trúc repo và lệnh chạy mong muốn

```text
T-024 (trong repo hiện tại)
├── README.md (hướng dẫn chuẩn bị và chạy)
├── requirements.txt hoặc pyproject.toml
├── src/
│   ├── download_data.py
│   ├── generate_synthetic.py
│   ├── adapters/{base.py,yolov8n.py,efficientnetv2_b0.py,convnext_tiny.py}
│   ├── benchmark.py
│   ├── evaluate.py
│   └── run_smoke_test.py
├── tests/
│   ├── test_metrics.py
│   ├── test_adapter_contract.py
│   └── test_manifest.py
├── data/                 # gitignored, ~ <=25 MB ảnh
├── checkpoints/          # gitignored
└── reports/              # summary / predictions / plots
```

Cho phép đặt project dưới `experiments/T024` nếu repo đã có cấu trúc khác; không đụng code cũ. Kỳ vọng một lệnh (hoặc tương đương):

```bash
python -m src.run_smoke_test --device auto --data-limit-mb 25 --seed 42 --threshold 0.5 --output reports/latest
```

Yêu cầu script tự thực hiện: check environment → tải/cached data → validate manifest → tạo synthetic → tải/cached weights → load 3 model → chạy đánh giá/latency → voting khi hợp lệ → xuất report. Script **idempotent** (chạy lại không tải trùng, không thay đổi input không cần thiết). Nếu một model lỗi vẫn kết thúc hai model còn lại, kèm file `errors.json` có traceback và status.

**Output bắt buộc:**

- `reports/latest/summary.csv`: model, dataset, n, TP, TN, FP, FN, accuracy, precision, recall, F1, FPR, p50/p95 end-to-end latency, model size, device, status.
- `reports/latest/predictions.csv`: đường dẫn/ID ảnh, true label, model, score, label, bbox nếu có, inference time.
- `reports/latest/results.json`: config, revisions/hashes, version, threshold, metrics và errors.
- `reports/latest/README.md`: bảng so sánh, kết luận model tối ưu theo tradeoff accuracy/speed, failure cases, hạn chế benchmark 122 ảnh, tính hợp lệ của voting.
- `reports/latest/failure_cases/`: ít nhất 5 ảnh false positive và 5 ảnh false negative nếu có, chọn theo confidence (không tự dựng ảnh lỗi).
- `tests/` có tối thiểu các unit test được chạy và passing; `git status` cho thấy data/weights không bị commit.

## 6. Milestones / Definition of Done

1. Khởi tạo branch riêng `feat/t024-watermark-smoke-test` nếu branch này chưa có, **không ghi đè thay đổi hiện hữu**; không force push, không commit/push nếu chưa được yêu cầu.
2. Data công khai tải và được kiểm tra đủ **122 ảnh**; synthetic/manifest nhất quán; image files trong budget.
3. Ba checkpoint tải từ đúng nguồn, hash/revisions có log; từng model chạy predict trên ít nhất 1 positive và 1 negative mà không crash.
4. Chạy đủ 122 ảnh/model, cộng synthetic nếu có; mô hình lỗi phải được ghi rõ (không tính là “đã benchmark đủ 3”).
5. Có bảng đo tốc độ cùng device và báo cáo nhầm lẫn, voting nếu thích hợp.
6. README ghi lệnh tái lập và kết luận có chứng cứ; báo cáo sự cố và kiến nghị bước tiếp theo (thêm independent text-watermark test set và hand-label).

## 7. PROMPT DÁN TRỰC TIẾP VÀO CODEX

> **Lưu ý:** nên mở workspace đúng repo muốn phát triển trước khi chạy prompt dưới đây. Đưa Codex file kế hoạch này vào thư mục làm việc (ví dụ `docs/T024_PLAN.md`) hoặc dán cả nội dung vào chat.

```text
Bạn là software/research engineer. Hãy TỰ CHỦ triển khai task T-024 watermark detection smoke test theo file T024_Watermark_Smoke_Test_Codex_Plan.md hiện có trong workspace (nếu file chưa có, dùng các thông số sau). Làm thật, chạy thật các lệnh và tạo file báo cáo, KHÔNG chỉ viết pseudocode hoặc tạo kết quả giả.

Mục tiêu: tải bộ dữ liệu boomb0om/watermarks-validation gồm 122 ảnh (62 positive, 60 negative, tổng ~12.6 MB) từ Hugging Face bằng snapshot_download với allow_patterns clean/**, watermark/**, validation_data.csv; sinh thêm tập synthetic nhỏ bằng PIL với chữ SAMPLE/PREVIEW/WATERMARK, không vượt 25 MB DỮ LIỆU ẢNH. KHÔNG tải full dataset hàng GB, không train.

Benchmark 3 checkpoint ĐÃ ĐƯỢC FINE-TUNE CHO WATERMARK:
1) qfisch/yolov8n-watermark-detection, best.pt (~6.23MB), YOLOv8n, ultralytics.
2) qwertyforce/watermark_detection, model.pt (~23.9MB), EfficientNetV2-B0, pickle checkpoint: kiểm tra tương thích timm và an toàn deserialize; score watermark đúng hướng sigmoid.
3) boomb0om/watermark-detectors, convnext-tiny_watermarks_detector.pth (~113MB), dùng kiến trúc từ boomb0om/watermark-detection, torchvision transforms 256x256/ImageNet normalize, logits index 1 = watermark.

Ưu tiên code module adapters tách biệt, tự detect CPU/CUDA, chạy hết pipeline một lệnh `python -m src.run_smoke_test --device auto --data-limit-mb 25 --seed 42 --output reports/latest`. Bảo vệ thay đổi người khác trong repo; tạo branch `feat/t024-watermark-smoke-test` nếu cần và an toàn. Đặt data và weights vào .gitignore. Chỉ tải từ các URL/repo được liệt kê, không tải thêm weights gốc ImageNet thay thế. Với checkpoint pickle, giữ trong môi trường cô lập, verify source/revision/checksum, không chạy mã bên ngoài; nếu cần cơ chế load không an toàn mà không được phép, ghi blocked rõ ràng, tuyệt đối không bịa số liệu.

Chạy batch=1; metrics accuracy, precision/recall/F1 watermark, FPR, confusion matrix, TP/TN/FP/FN, p50/p95 latency end-to-end ms/ảnh, throughput, checkpoint size, device; warmup rồi timing, torch.cuda.synchronize() trên GPU. Ghi predictions từng ảnh và failure cases; test public_122 và synthetic riêng. Dùng threshold 0.5 chung, YOLO score = max confidence bbox của watermark; ConvNeXt softmax index 1; EfficientNet kiểm tra sigmoid đầu ra và nhãn. Khi 3 model chạy thành công, chạy thêm majority vote và so sánh accuracy vs latency.

Sinh README, script download/benchmark/evaluate, unit tests, summary.csv, predictions.csv, results.json, failure gallery và báo cáo markdown. Thử nghiệm thật, sửa lỗi cần thiết, chạy tests. Nếu bị chặn vì mạng/thiếu quyền/phần cứng, dừng đúng phần bị chặn, làm tiếp phần độc lập, báo nguyên nhân cụ thể và lệnh người dùng phải làm; không báo đã thành công khi chưa chạy. Cuối cùng gửi summary dạng bảng, các path artifact, device, lỗi/giới hạn, cùng đúng command để tôi reproduce. Không commit hoặc push nếu chưa được yêu cầu.
```

## 8. Sources / kiểm chứng

- YOLO Nano model, mục tiêu text/logo, license: https://huggingface.co/qfisch/yolov8n-watermark-detection
- YOLO Nano file `best.pt`: https://huggingface.co/qfisch/yolov8n-watermark-detection/tree/main
- EfficientNetV2-B0 model card và `model.pt`: https://huggingface.co/qwertyforce/watermark_detection
- ConvNeXt code, API và thử nghiệm gốc: https://github.com/boomb0om/watermark-detection
- ConvNeXt weight: https://huggingface.co/boomb0om/watermark-detectors
- Dataset, 122 ảnh và CSV nhãn: https://huggingface.co/datasets/boomb0om/watermarks-validation
- Dự án fine-tune YOLO và nhóm mô hình tương tự: https://github.com/tgenlis83/dnn-watermark

**Trạng thái nghiên cứu:** các nguồn, kích thước checkpoint và dataset đã kiểm tra trực tuyến tại thời điểm lập kế hoạch. **Chưa thực thi download/inference trong workspace của bạn**, nên không có benchmark đo thực nghiệm do mình đưa ra.
