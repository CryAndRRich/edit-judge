# T-051 · Thử nghiệm model chỉnh màu

So sánh InstructPix2Pix và MagicBrush trên 5 ảnh đường phố. Bài thử dùng lệnh
`Make the scene look like it was photographed at sunset` từ T-027 để đánh giá
đổi tông màu và ánh sáng toàn cảnh.

- [Kết quả và nhận xét](results/REPORT.md)
- [Danh sách ảnh và SHA-256](sample.json)
- [Nguồn bộ dữ liệu EditJudge](https://www.kaggle.com/datasets/cryandrrich/editjudge?select=images)

## Model

| Model | Checkpoint |
| --- | --- |
| InstructPix2Pix | [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix/tree/31519b5cb02a7fd89b906d88731cd4d6a7bbf88d) |
| MagicBrush | [vinesmsuic/magicbrush-jul7](https://huggingface.co/vinesmsuic/magicbrush-jul7/tree/d75a9f18439f67dbdc0ab113036f491ce2256c53), bản Diffusers được tác giả [dẫn trong repo MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush#load-in-diffusers) |

Cả hai model dùng FP16, CPU offload, Euler Ancestral, seed 42, 30 bước,
text guidance 7.5 và image guidance 1.5. Ảnh gốc 150 × 150 được đưa về
144 × 144, làm tròn xuống theo bội số của 8.

## Chạy lại

Chạy các lệnh từ thư mục này. Cần Python 3.10 trở lên và PyTorch 2.6 trở lên
có CUDA để sinh ảnh. Tạo môi trường riêng, cài PyTorch phù hợp với GPU trước, rồi cài
`requirements.txt`. [Hướng dẫn cài PyTorch](https://pytorch.org/get-started/locally/)

```powershell
python -m pip install -r requirements.txt
python benchmark.py run --images ../.. --model instruct-pix2pix --output outputs/rerun-pix2pix
python benchmark.py run --images ../.. --model magicbrush --output outputs/rerun-magicbrush
python benchmark.py compare --images ../.. --edit instruct-pix2pix=outputs/rerun-pix2pix/images --edit magicbrush=outputs/rerun-magicbrush/images --report-dir outputs/rerun-metrics --preview outputs/rerun-comparison.png
```

`../..` trỏ đến 5 ảnh và hai ZIP hiện có trong `D:\DS`. Có thể truyền thư mục
ảnh khác qua `--images`. Trên Kaggle, gắn bộ dữ liệu vào notebook, bật GPU và
Internet, rồi dùng đường dẫn ảnh dưới `/kaggle/input` và ghi kết quả xuống
`/kaggle/working`. Script kiểm tra đủ 5 ảnh bằng SHA-256 trước khi tải model.
Mỗi lần chạy dùng một thư mục đầu ra mới, có `run.json`, `run.csv` và `images/`.
Có thể dùng `--checkpoint` với một thư mục Diffusers đã tải sẵn.

Tạo lại bảng và ảnh đối chiếu của hai ZIP có sẵn, có thể chạy trên CPU.

```powershell
python benchmark.py compare --images ../.. --edit instruct-pix2pix=../../instruct-pix2pix.zip --edit magicbrush=../../magicbrush.zip --report-dir outputs/archive-metrics --preview outputs/archived-comparison.png
```

Khi chuẩn bị commit, chạy `git add -- t051-coloring/` từ gốc repo để thêm mã,
cấu hình và báo cáo. `data/`, `outputs/`, checkpoint và ZIP được loại khỏi Git
qua `.gitignore`.
