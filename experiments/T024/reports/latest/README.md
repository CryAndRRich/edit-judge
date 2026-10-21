# T-024: Kết quả smoke test thực chạy

Run: `2026-10-18T22:11:31.785845+07:00` → `2026-10-18T22:12:00.734500+07:00` (Asia/Bangkok).
Device: **cuda:0**, NVIDIA GeForce RTX 4050 Laptop GPU; CPU: AMD Ryzen 5 7235HS; threads=4; batch=1; FP32.

Threshold chung **0.5**; warmup **8** lượt/model, không tính vào latency. Không training/fine-tuning.

## So sánh

### public_122

| Model | n | Accuracy | Precision | Recall | F1 | FPR | E2E mean / p50 / p95 (ms) | ảnh/s | Checkpoint MB | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| yolov8n | 122 | 58.20% | 82.35% | 22.58% | 35.44% | 5.00% | 45.11 / 34.72 / 107.57 | 22.17 | 6.229 | ok |
| efficientnetv2_b0 | 122 | 60.66% | 76.92% | 32.26% | 45.45% | 10.00% | 28.12 / 26.73 / 37.09 | 35.57 | 23.870 | ok |
| convnext_tiny | 122 | 93.44% | 98.21% | 88.71% | 93.22% | 1.67% | 16.97 / 15.51 / 25.96 | 58.94 | 113.446 | ok |
| ensemble_voting | 122 | 70.49% | 93.33% | 45.16% | 60.87% | 3.33% | 90.19 / 77.48 / 154.58 | 11.09 | 143.545 | ok |

| Model | TN | FP | FN | TP | Model-only mean / p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|
| yolov8n | 57 | 3 | 48 | 14 | 21.95 / 13.94 / 80.47 |
| efficientnetv2_b0 | 54 | 6 | 42 | 20 | 22.78 / 22.54 / 27.06 |
| convnext_tiny | 59 | 1 | 7 | 55 | 12.62 / 12.36 / 15.57 |
| ensemble_voting | 58 | 2 | 34 | 28 | 57.35 / 49.52 / 113.29 |

### synthetic

| Model | n | Accuracy | Precision | Recall | F1 | FPR | E2E mean / p50 / p95 (ms) | ảnh/s | Checkpoint MB | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| yolov8n | 60 | 38.33% | 100.00% | 7.50% | 13.95% | 0.00% | 29.64 / 27.14 / 50.30 | 33.73 | 6.229 | ok |
| efficientnetv2_b0 | 60 | 40.00% | 100.00% | 10.00% | 18.18% | 0.00% | 28.23 / 27.75 / 35.22 | 35.42 | 23.870 | ok |
| convnext_tiny | 60 | 66.67% | 95.45% | 52.50% | 67.74% | 5.00% | 16.03 / 15.59 / 20.15 | 62.40 | 113.446 | ok |
| ensemble_voting | 60 | 41.67% | 100.00% | 12.50% | 22.22% | 0.00% | 73.90 / 72.16 / 94.88 | 13.53 | 143.545 | ok |

| Model | TN | FP | FN | TP | Model-only mean / p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|
| yolov8n | 20 | 0 | 37 | 3 | 11.26 / 10.78 / 15.30 |
| efficientnetv2_b0 | 20 | 0 | 36 | 4 | 23.90 / 23.49 / 28.97 |
| convnext_tiny | 19 | 1 | 19 | 21 | 12.73 / 12.27 / 16.53 |
| ensemble_voting | 20 | 0 | 35 | 5 | 47.88 / 47.38 / 53.85 |

## Nhận xét từ số đo

- F1 cao nhất trên public_122: **convnext_tiny** (0.9322); model đơn nhanh nhất: **convnext_tiny** (16.97 ms/ảnh).
- Recall cao nhất: **convnext_tiny**, 55/62 watermark; FPR 1/60 clean. Khi lọc ảnh, cần cân nhắc riêng watermark bỏ sót và clean loại nhầm.
- Voting: Accuracy 0.7049, F1 0.6087, recall 0.4516, FPR 0.0333; chênh F1 so model đơn tốt nhất -0.3235; chi phí tuần tự 90.19 ms/ảnh (~5.32× model đó). Không cải thiện F1 để bù chi phí chạy ba model trên tập này.

## Phạm vi latency và tính hợp lệ

- E2E đo bằng perf_counter: EXIF transpose/RGB, tiền xử lý, chuyển tensor lên thiết bị, forward và hậu xử lý/score/box. Ảnh đã decode bằng PIL ngoài timer; không bao gồm I/O đĩa, download, load model hoặc warmup. CUDA synchronize trước/sau.
- Model-only classifier bao gồm forward trên tensor có sẵn ở thiết bị; YOLO dùng result.speed[inference] do thư viện đo. Không gọi tốc độ nội bộ YOLO là E2E. Throughput = 1000 / mean latency ở batch=1, không phải benchmark batch hoặc phục vụ đồng thời.
- Ensemble majority >=2/3 chỉ được tạo khi cả ba model có toàn bộ ID ảnh hợp lệ. Latency voting là **tổng các thời gian adapter đã đo trên từng ảnh**, ước lượng chạy tuần tự; chưa đo một pipeline ensemble riêng. Checkpoint voting là tổng ba file, không có checkpoint thứ tư.

## Dữ liệu và hạn chế

- Public: {'repo': 'boomb0om/watermarks-validation', 'revision': 'd2d22701e18e96abd28fc476e9396501adf87123', 'image_bytes': 12562610, 'csv_sha256': '5b627a89cfe2caeaa896ca836c382e3ac355f1cc0c421944817123d01027f9e4', 'download_or_cache_s': 1.094020200005616}. Manifest đầy đủ kèm SHA-256 ở `public_manifest.json`.
- Synthetic: {'status': 'ok', 'n': 60, 'positive': 40, 'negative': 20, 'combined_image_bytes': 14426408, 'font': 'Arial Bold, C:/Windows/Fonts/arialbd.ttf', 'font_sha256': 'e8f4e3baf6cc35fed6fcce3a540e8b39e8f6cda1d22a28f2ec8f526fef7a43f5', 'negative_policy': 'Unchanged clean backgrounds, JPEG quality 80; no semantic overlays'}. 20 nền clean từ public tạo 40 watermark chữ và 20 negative không overlay; không độc lập với public. Xem [contact sheet](synthetic_contact_sheet.jpg) và `synthetic_manifest.csv`.
- Public_122 là tập đánh giá tác giả ConvNeXt từng dùng; có thể thiên vị model này. 122 ảnh quá ít để kết luận generalization, không có khoảng tin cậy hiệu năng triển khai. Synthetic kế thừa nhãn clean của nguồn và chưa có người hand-review độc lập; không gộp accuracy với public.
- EfficientNet: model card không công bố transform fine-tune; baseline dùng cấu hình timm nhúng trong checkpoint (192px center crop, bicubic, ImageNet normalize). Cấu hình này có thể bỏ watermark ở rìa; chưa được xác nhận là transform đánh giá của tác giả. Không dò transform/threshold bằng test set.
- Khuyến nghị: thêm tập text-watermark độc lập, người gán nhãn kiểm tra watermark mờ/góc/chéo và đánh giá với chính sách FPR/recall của sản phẩm.

## Checkpoint và tương thích

- Tải đúng repo/file allowlist, pin commit lần đầu trong revisions.json, đối chiếu SHA-256 file với LFS từ Hugging Face. Không trust_remote_code, torch.hub, pretrained ImageNet thay thế, hoặc script từ nguồn ngoài.
- Cả ba checkpoint nạp bằng **weights_only=True**. ConvNeXt strict state_dict; EfficientNet/YOLO dùng safe_globals chứa các lớp cụ thể đã kiểm tra. YOLO thay loader mặc định để ngăn weights_only=False và auto-install; dill._load_type được thay bằng hàm cục bộ chỉ chấp nhận set. Đây là giảm rủi ro pickle, không phải sandbox bảo mật.
- Venv riêng `.venv-t024` kế thừa gói nền qua --system-site-packages để dùng PyTorch CUDA sẵn có; gói bổ sung cài trong venv, không sửa môi trường bài học Python 3.8. Venv không phải rào chắn bảo mật hệ điều hành.
- timm 0.6.13 không có import paths timm.layers / timm.models._efficientnet_blocks được checkpoint lưu; chuyển sang **0.9.16**. YOLO names thực tế là logo/text; ánh xạ cả hai theo model card qfisch. EfficientNet có Linear(1280,1), sigmoid đúng một lần. ConvNeXt có head 768→512→256→2, softmax index 1.

### yolov8n

Status: `ok`; ảnh thành công: 182.
- Repo/file: `qfisch/yolov8n-watermark-detection/best.pt`; commit `26ce92a7e777ae955247e5c4a74ee3ea5279924a`.
- SHA-256: `f40b75a88079ba9e9dfba3d2c279092a5264a4e36bf3c645ab29e466775c7317`; bytes=6228505; download/cache=0.009s; load=0.363s.
```json
{
  "classes": {
    "0": "logo",
    "1": "text"
  },
  "watermark_classes": [
    0,
    1
  ],
  "conf": 0.001,
  "imgsz": 640,
  "class_mapping_evidence": "qfisch model card: watermark detection of both texts and logos; exact checkpoint names logo/text",
  "preprocess": "PIL RGB + EXIF transpose; Ultralytics letterbox imgsz=640, scale 1/255, no ImageNet normalization",
  "score": "max confidence over watermark boxes, default 0",
  "globals": [
    "dill._dill._load_type",
    "torch.nn.modules.activation.SiLU",
    "torch.nn.modules.batchnorm.BatchNorm2d",
    "torch.nn.modules.container.ModuleList",
    "torch.nn.modules.container.Sequential",
    "torch.nn.modules.conv.Conv2d",
    "torch.nn.modules.loss.BCEWithLogitsLoss",
    "torch.nn.modules.pooling.MaxPool2d",
    "torch.nn.modules.upsampling.Upsample",
    "ultralytics.nn.modules.block.Bottleneck",
    "ultralytics.nn.modules.block.C2f",
    "ultralytics.nn.modules.block.DFL",
    "ultralytics.nn.modules.block.SPPF",
    "ultralytics.nn.modules.conv.Concat",
    "ultralytics.nn.modules.conv.Conv",
    "ultralytics.nn.modules.head.Detect",
    "ultralytics.nn.tasks.DetectionModel",
    "ultralytics.utils.IterableSimpleNamespace",
    "ultralytics.utils.loss.BboxLoss",
    "ultralytics.utils.loss.v8DetectionLoss",
    "ultralytics.utils.tal.TaskAlignedAssigner"
  ],
  "load_policy": "weights_only=True with explicit allowlist and restricted dill set type shim; stock loader replaced",
  "model_only_timing": "Ultralytics result.speed[inference] forward timing; distinct from adapter end-to-end"
}
```

### efficientnetv2_b0

Status: `ok`; ảnh thành công: 182.
- Repo/file: `qwertyforce/watermark_detection/model.pt`; commit `6114db333950e5d26376961cd9a7a463ea602215`.
- SHA-256: `c835dcf80ffa6fd3cb2cedfba95223b5c4243bf59431dfee5889517016a8dff5`; bytes=23870125; download/cache=0.028s; load=0.638s.
```json
{
  "architecture": "timm.models.efficientnet.EfficientNet",
  "classifier": "Linear(in_features=1280, out_features=1, bias=True)",
  "output": "one raw logit; apply sigmoid exactly once",
  "score": "sigmoid(logit); watermark=1 per model card",
  "preprocess": "Compose(\n    Resize(size=219, interpolation=bicubic, max_size=None, antialias=True)\n    CenterCrop(size=(192, 192))\n    ToTensor()\n    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))\n)",
  "data_config": {
    "input_size": [
      3,
      192,
      192
    ],
    "interpolation": "bicubic",
    "mean": [
      0.485,
      0.456,
      0.406
    ],
    "std": [
      0.229,
      0.224,
      0.225
    ],
    "crop_pct": 0.875,
    "crop_mode": "center"
  },
  "embedded_config": {
    "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-effv2-weights/tf_efficientnetv2_b0-c7cc451f.pth",
    "hf_hub_id": "timm/tf_efficientnetv2_b0.in1k",
    "architecture": "tf_efficientnetv2_b0",
    "tag": "in1k",
    "custom_load": false,
    "input_size": [
      3,
      192,
      192
    ],
    "test_input_size": [
      3,
      224,
      224
    ],
    "fixed_input_size": false,
    "interpolation": "bicubic",
    "crop_pct": 0.875,
    "crop_mode": "center",
    "mean": [
      0.485,
      0.456,
      0.406
    ],
    "std": [
      0.229,
      0.224,
      0.225
    ],
    "num_classes": 1000,
    "pool_size": [
      6,
      6
    ],
    "first_conv": "conv_stem",
    "classifier": "classifier"
  },
  "globals": [
    "timm.layers.adaptive_avgmax_pool.SelectAdaptivePool2d",
    "timm.layers.conv2d_same.Conv2dSame",
    "timm.layers.drop.DropPath",
    "timm.layers.norm_act.BatchNormAct2d",
    "timm.models._efficientnet_blocks.ConvBnAct",
    "timm.models._efficientnet_blocks.EdgeResidual",
    "timm.models._efficientnet_blocks.InvertedResidual",
    "timm.models._efficientnet_blocks.SqueezeExcite",
    "timm.models.efficientnet.EfficientNet",
    "torch.nn.modules.activation.SiLU",
    "torch.nn.modules.activation.Sigmoid",
    "torch.nn.modules.container.Sequential",
    "torch.nn.modules.conv.Conv2d",
    "torch.nn.modules.flatten.Flatten",
    "torch.nn.modules.linear.Identity",
    "torch.nn.modules.linear.Linear",
    "torch.nn.modules.pooling.AdaptiveAvgPool2d"
  ],
  "load_policy": "weights_only=True with explicit audited class allowlist",
  "mapping_note": "Author card specifies watermark probability via sigmoid; examples logged without choosing orientation or threshold"
}
```

### convnext_tiny

Status: `ok`; ảnh thành công: 182.
- Repo/file: `boomb0om/watermark-detectors/convnext-tiny_watermarks_detector.pth`; commit `bd472165157f6d14ddcd3b24846ffd828e8fee8c`.
- SHA-256: `a83262f23f4fc6e87b0043a3a766e2c6eabcce63e26af78f35d5683d33066866`; bytes=113445965; download/cache=0.144s; load=0.311s.
```json
{
  "architecture": "ConvNeXt tiny [3,3,9,3], custom 768/512/256/2 head",
  "preprocess": "RGB, EXIF transpose, Resize(256,256) bilinear, ImageNet normalize",
  "score": "softmax(logits)[1]",
  "fp16": false,
  "load_policy": "weights_only=True, strict state_dict"
}
```

### ensemble_voting

Status: `ok`; ảnh thành công: 182.
- Tổng checkpoint ba model: 143544595 bytes. Sum of measured per-image adapter times; sequential estimate, not separately timed voting pipeline

## Failure cases và sự cố

- [Manifest ảnh lỗi](failure_cases/manifest.csv): tối đa 5 FP confidence cao nhất và 5 FN confidence thấp nhất mỗi model; preview thu nhỏ từ ảnh thật, giữ path và hash ảnh gốc. Nếu ít hơn 5 thì xuất toàn bộ lỗi hiện có.
- Lỗi chưa giải quyết của lần chạy này: **0**. Xem [errors.json](errors.json); lịch sử kiểm tra tương thích ở inspection.log và compatibility_notes.md.

## Tái lập

Từ thư mục `experiments/T024` trong repository, PowerShell:

```powershell
.\.venv-t024\Scripts\python.exe -m src.run_smoke_test --device auto --data-limit-mb 25 --seed 42 --threshold 0.5 --threads 4 --warmup 8 --output reports/latest
.\.venv-t024\Scripts\python.exe -m pytest -q tests
```

Revisions được giữ cố định khi chạy lại; input cached không ghi lại nếu không đổi. Xem [README project](../../README.md), requirements.txt và requirements-t024-lock.txt. Phiên bản/phần cứng đầy đủ nằm trong [results.json](results.json).

Ghi chú lịch sử: tại thời điểm benchmark 18/10/2026, workspace chỉ có branch cục bộ `feat/t024-watermark-smoke-test`, chưa commit/push. Bản công bố tài liệu và kết quả hiện nằm trong `experiments/T024` trên branch `watermark-model`; data/checkpoints/venv bị ignore.

Nguồn: [YOLO](https://huggingface.co/qfisch/yolov8n-watermark-detection), [EfficientNet](https://huggingface.co/qwertyforce/watermark_detection), [ConvNeXt code](https://github.com/boomb0om/watermark-detection), [weights](https://huggingface.co/boomb0om/watermark-detectors), [dataset](https://huggingface.co/datasets/boomb0om/watermarks-validation). YOLO AGPL-3.0 và tác giả ghi academic usage only; EfficientNet CC BY-SA 4.0; ConvNeXt HF repo BSD-3-Clause.
