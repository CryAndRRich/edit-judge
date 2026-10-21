# T-024 — Watermark smoke test

Pipeline phát hiện watermark nhìn thấy, chạy inference bằng đúng ba checkpoint trong kế hoạch. Không training, không tải dataset lớn. Báo cáo thực chạy: [reports/latest/README.md](reports/latest/README.md).

## Môi trường đã dùng

Windows 11, Python 3.12.10, PyTorch 2.6.0+cu124 / torchvision 0.21.0+cu124; RTX 4050 Laptop 6 GB. Workspace có môi trường riêng `.venv-t024`; gói nền PyTorch được kế thừa bằng `--system-site-packages` để tránh tải lại CUDA. Các dependency bổ sung nằm trong venv; môi trường Python 3.8 của bài học được giữ nguyên.

Chạy lại trong workspace hiện tại bằng PowerShell:

```powershell
.\.venv-t024\Scripts\python.exe -m src.run_smoke_test --device auto --data-limit-mb 25 --seed 42 --threshold 0.5 --threads 4 --warmup 8 --output reports/latest
.\.venv-t024\Scripts\python.exe -m pytest -q tests
.\.venv-t024\Scripts\python.exe -m pip check
```

Tạo môi trường sạch trên máy khác (Python 3.12 đã cài):

```powershell
py -3.12 -m venv .venv-t024
.\.venv-t024\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv-t024\Scripts\python.exe -m pip install -r requirements-t024-lock.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

Không có CUDA thì dùng PyTorch CPU và `--device cpu`; benchmark lúc đó phải báo riêng, không so latency trực tiếp với kết quả GPU. `requirements.txt` liệt kê dependency trực tiếp; `requirements-t024-lock.txt` pin dependency closure thực tế của pipeline; `reports/latest/pip-freeze.txt` là toàn bộ inventory venv và gói nền được kế thừa.

## Pipeline và đầu ra

`src.run_smoke_test` tự kiểm tra môi trường, tải/cached dataset, xác minh đúng 122 ảnh và nhãn 62/60, tạo synthetic seed 42, tải và kiểm hash ba checkpoint, nạp/chạy từng model, đánh giá và tạo báo cáo. Chạy lại dùng revision lần đầu trong `reports/latest/revisions.json`; giữ file này khi tái lập. Chỉ xóa/đổi file pin khi chủ động muốn thay revision. Không dùng threshold tối ưu chọn từ test set.

- `src/download_data.py`: snapshot allowlist `clean/**`, `watermark/**`, `validation_data.csv`; manifest, kích thước và SHA-256.
- `src/generate_synthetic.py`: 20 nền clean tạo 40 positive có SAMPLE/PREVIEW/WATERMARK và 20 negative không overlay; JPEG quality 80; tổng ảnh nguồn + synthetic ≤25 MB (decimal MB). Synthetic sử dụng Arial Bold của Windows, hash font được lưu.
- `src/adapters/`: YOLO imgsz640/conf0.001/max box; EfficientNet một logit/sigmoid với cấu hình timm nhúng 192px center crop; ConvNeXt256/ImageNet normalize/softmax index1.
- `src/benchmark.py`, `src/evaluate.py`: batch1, warmup8, synchronize CUDA; latency E2E không bao gồm decode ảnh từ đĩa. Model-only tách riêng. Ensemble majority chỉ khi cả ba có đủ ảnh, latency = tổng các thời gian đã đo, không phải pipeline ensemble đo riêng.
- `reports/latest/`: summary.csv, predictions.csv, results.json, errors.json, README.md, confusion SVG, synthetic contact sheet/manifest và failure_cases/manifest.csv kèm preview ảnh thật. Số liệu public/synthetic tách riêng.
- `tests/`: confusion/metrics/percentile, contract score/threshold, manifest/path/nhãn và restricted pickle shim. `pytest.ini` đặt temp trong workspace để tránh thư mục temp hệ thống bị từ chối quyền.

Nếu một adapter lỗi, adapter còn lại vẫn chạy; traceback nằm trong errors.json. Chương trình trả exit code 1 khi chưa chạy đủ cả ba model; không bù dữ liệu thiếu hoặc tạo số liệu giả. `*_run.json` giữ kết quả từng model ngay sau khi chạy xong.

## An toàn checkpoint và giới hạn

Nguồn, filename và commit được pin; SHA-256 checkpoint đối chiếu với metadata LFS từ Hugging Face trước khi nạp. ConvNeXt dùng `weights_only=True` và strict state_dict. EfficientNet/YOLO cũng dùng `weights_only=True` với danh sách lớp cụ thể đã kiểm tra; unknown global bị chặn. Không gọi `weights_only=False`, `trust_remote_code`, `torch.hub`, pretrained ImageNet hoặc auto-install từ checkpoint. YOLO stock loader được thay bằng loader cục bộ; hàm dill được thay bằng shim chỉ chấp nhận kiểu set. Venv và weights-only giảm rủi ro nhưng không phải sandbox hệ điều hành.

Public_122 từng được tác giả ConvNeXt dùng; kết quả không chứng minh generalization độc lập. Synthetic trùng nền public, có chữ rất mờ, kế thừa nhãn clean từ nguồn và chưa được người gán nhãn độc lập kiểm tra. EfficientNet model card không công bố preprocessing fine-tune; baseline dùng cấu hình có sẵn trong checkpoint và ghi rõ trong report. Không suy diễn kết quả sang watermark ẩn hoặc xóa watermark.

Tài liệu, mã nguồn và kết quả T-024 được lưu trong `experiments/T024` trên branch `watermark-model` của [CryAndRRich/edit-judge](https://github.com/CryAndRRich/edit-judge/tree/watermark-model). Chạy các lệnh ở trên từ thư mục `experiments/T024`. `.gitignore` loại data/checkpoints/venv/cache; pipeline tự tải lại dữ liệu và checkpoint theo revision đã pin. Các log và số liệu trong `reports/latest` được giữ từ lần benchmark ngày 18/10/2026; `git-status.txt` ghi trạng thái workspace tại thời điểm đó.

Nguồn: [YOLO Nano](https://huggingface.co/qfisch/yolov8n-watermark-detection), [EfficientNetV2](https://huggingface.co/qwertyforce/watermark_detection), [ConvNeXt checkpoint](https://huggingface.co/boomb0om/watermark-detectors), [ConvNeXt code](https://github.com/boomb0om/watermark-detection), [dataset](https://huggingface.co/datasets/boomb0om/watermarks-validation). Bản nguồn đã đọc nằm trong reports/latest/sources; không chạy script tải từ repo.
