"""Generate reviewable reports solely from actual predictions and timings."""
import csv
import json
from pathlib import Path

from PIL import Image, ImageOps
from .benchmark import write_csv
from .download_data import sha256, write_json


def failure_gallery(predictions, output):
    root = output / 'failure_cases'
    root.mkdir(parents=True, exist_ok=True)
    # Remove only this generator's old previews so a failed rerun cannot expose stale cases.
    for name in ('yolov8n', 'efficientnetv2_b0', 'convnext_tiny', 'ensemble_voting'):
        for kind in ('FP', 'FN'):
            for rank in range(1, 6):
                (root/f'{name}_{kind}_{rank}.jpg').unlink(missing_ok=True)
    manifest = []
    for model in dict.fromkeys(p['model'] for p in predictions):
        for kind, label in [('FP', 0), ('FN', 1)]:
            cases = [p for p in predictions if p['model'] == model and p['true_label'] == label and p['label'] != label]
            cases.sort(key=lambda p: p['score'], reverse=kind == 'FP')
            for i, p in enumerate(cases[:5]):
                path = root / f'{model}_{kind}_{i+1}.jpg'
                with Image.open(p['path']) as im:
                    im = ImageOps.exif_transpose(im).convert('RGB')
                    im.thumbnail((640, 640)); im.save(path, quality=80)
                manifest.append(dict(model=model, type=kind, rank=i+1, id=p['id'], source=p['path'],
                                     score=p['score'], true_label=p['true_label'], predicted_label=p['label'],
                                     source_sha256=sha256(p['path']), preview=path.name))
    write_csv(root / 'manifest.csv', manifest)
    return manifest


def confusion_svg(summary, output):
    for row in summary:
        if 'TP' not in row:
            continue
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="440" height="300">',
                 '<rect width="440" height="300" fill="white"/>',
                 f'<text x="20" y="25" font-family="sans-serif" font-size="16">{row["model"]} / {row["dataset"]}</text>',
                 '<text x="170" y="60" font-family="sans-serif">Pred clean</text><text x="290" y="60" font-family="sans-serif">Pred WM</text>']
        for j, (key, x, y, color) in enumerate([('TN',150,80,'#ccebdc'),('FP',270,80,'#f5d0c5'),('FN',150,180,'#f5d0c5'),('TP',270,180,'#ccebdc')]):
            parts += [f'<rect x="{x}" y="{y}" width="110" height="90" fill="{color}"/>',
                      f'<text x="{x+15}" y="{y+50}" font-size="22" font-family="sans-serif">{key} {row[key]}</text>']
        parts += ['<text x="10" y="130" font-family="sans-serif">True clean</text><text x="10" y="230" font-family="sans-serif">True WM</text>', '</svg>']
        (output / f'confusion_{row["model"]}_{row["dataset"]}.svg').write_text('\n'.join(parts), encoding='utf-8')


def markdown_report(results, output):
    summaries = results['summary']
    env, cfg = results['environment'], results['config']
    lines = ['# T-024: Kết quả smoke test thực chạy', '',
             f"Run: `{results['started_at']}` → `{results['finished_at']}` (Asia/Bangkok).",
             f"Device: **{cfg['resolved_device']}**, {env.get('gpu', 'CPU')}; CPU: {env.get('cpu', '')}; threads={cfg['threads']}; batch=1; FP32.", '',
             f"Threshold chung **{cfg['threshold']}**; warmup **{cfg['warmup']}** lượt/model, không tính vào latency. Không training/fine-tuning.", '',
             '## So sánh', '']
    for dataset in dict.fromkeys(s['dataset'] for s in summaries):
        lines += [f'### {dataset}', '',
                  '| Model | n | Accuracy | Precision | Recall | F1 | FPR | E2E mean / p50 / p95 (ms) | ảnh/s | Checkpoint MB | Status |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for s in (x for x in summaries if x['dataset'] == dataset):
            val = lambda k: f'{100*s[k]:.2f}%' if k in s else '—'
            times = ' / '.join(f'{s[k]:.2f}' for k in ('e2e_mean_ms', 'e2e_p50_ms', 'e2e_p95_ms')) if 'e2e_mean_ms' in s else '—'
            speed = f'{s["e2e_throughput_images_s"]:.2f}' if 'e2e_throughput_images_s' in s else '—'
            lines += [f"| {s['model']} | {s['n']} | {val('accuracy')} | {val('precision')} | {val('recall')} | {val('F1')} | {val('FPR')} | {times} | {speed} | {s['model_bytes']/1e6:.3f} | {s['status']} |"]
        lines += ['', '| Model | TN | FP | FN | TP | Model-only mean / p50 / p95 (ms) |', '|---|---:|---:|---:|---:|---:|']
        for s in (x for x in summaries if x['dataset'] == dataset and 'TP' in x):
            times = ' / '.join(f'{s[k]:.2f}' for k in ('model_mean_ms', 'model_p50_ms', 'model_p95_ms')) if 'model_mean_ms' in s else '—'
            lines += [f"| {s['model']} | {s['TN']} | {s['FP']} | {s['FN']} | {s['TP']} | {times} |"]
        lines += ['']
    public = [s for s in summaries if s['dataset'] == 'public_122' and s['status'] == 'ok' and s['model'] != 'ensemble_voting']
    lines += ['## Nhận xét từ số đo', '']
    if public:
        quality = max(public, key=lambda s: s['F1'])
        fast = min(public, key=lambda s: s['e2e_mean_ms'])
        recall = max(public, key=lambda s: s['recall'])
        lines += [f"- F1 cao nhất trên public_122: **{quality['model']}** ({quality['F1']:.4f}); model đơn nhanh nhất: **{fast['model']}** ({fast['e2e_mean_ms']:.2f} ms/ảnh).",
                  f"- Recall cao nhất: **{recall['model']}**, {recall['TP']}/62 watermark; FPR {recall['FP']}/60 clean. Khi lọc ảnh, cần cân nhắc riêng watermark bỏ sót và clean loại nhầm."]
        vote = next((s for s in summaries if s['model'] == 'ensemble_voting' and s['dataset'] == 'public_122'), None)
        if vote:
            lines += [f"- Voting: Accuracy {vote['accuracy']:.4f}, F1 {vote['F1']:.4f}, recall {vote['recall']:.4f}, FPR {vote['FPR']:.4f}; chênh F1 so model đơn tốt nhất {vote['F1']-quality['F1']:+.4f}; chi phí tuần tự {vote['e2e_mean_ms']:.2f} ms/ảnh (~{vote['e2e_mean_ms']/quality['e2e_mean_ms']:.2f}× model đó). " +
                      ('Không cải thiện F1 để bù chi phí chạy ba model trên tập này.' if vote['F1'] <= quality['F1'] else 'Có cải thiện F1 trên tập này; cần kiểm chứng độc lập trước khi đánh đổi tốc độ.')]
    lines += ['', '## Phạm vi latency và tính hợp lệ', '',
              '- E2E đo bằng perf_counter: EXIF transpose/RGB, tiền xử lý, chuyển tensor lên thiết bị, forward và hậu xử lý/score/box. Ảnh đã decode bằng PIL ngoài timer; không bao gồm I/O đĩa, download, load model hoặc warmup. CUDA synchronize trước/sau.',
              '- Model-only classifier bao gồm forward trên tensor có sẵn ở thiết bị; YOLO dùng result.speed[inference] do thư viện đo. Không gọi tốc độ nội bộ YOLO là E2E. Throughput = 1000 / mean latency ở batch=1, không phải benchmark batch hoặc phục vụ đồng thời.',
              '- Ensemble majority >=2/3 chỉ được tạo khi cả ba model có toàn bộ ID ảnh hợp lệ. Latency voting là **tổng các thời gian adapter đã đo trên từng ảnh**, ước lượng chạy tuần tự; chưa đo một pipeline ensemble riêng. Checkpoint voting là tổng ba file, không có checkpoint thứ tư.', '',
              '## Dữ liệu và hạn chế', '',
              f"- Public: {results.get('dataset', {})}. Manifest đầy đủ kèm SHA-256 ở `public_manifest.json`.",
              f"- Synthetic: {results.get('synthetic', {})}. 20 nền clean từ public tạo 40 watermark chữ và 20 negative không overlay; không độc lập với public. Xem [contact sheet](synthetic_contact_sheet.jpg) và `synthetic_manifest.csv`.",
              '- Public_122 là tập đánh giá tác giả ConvNeXt từng dùng; có thể thiên vị model này. 122 ảnh quá ít để kết luận generalization, không có khoảng tin cậy hiệu năng triển khai. Synthetic kế thừa nhãn clean của nguồn và chưa có người hand-review độc lập; không gộp accuracy với public.',
              '- EfficientNet: model card không công bố transform fine-tune; baseline dùng cấu hình timm nhúng trong checkpoint (192px center crop, bicubic, ImageNet normalize). Cấu hình này có thể bỏ watermark ở rìa; chưa được xác nhận là transform đánh giá của tác giả. Không dò transform/threshold bằng test set.',
              '- Khuyến nghị: thêm tập text-watermark độc lập, người gán nhãn kiểm tra watermark mờ/góc/chéo và đánh giá với chính sách FPR/recall của sản phẩm.', '',
              '## Checkpoint và tương thích', '',
              '- Tải đúng repo/file allowlist, pin commit lần đầu trong revisions.json, đối chiếu SHA-256 file với LFS từ Hugging Face. Không trust_remote_code, torch.hub, pretrained ImageNet thay thế, hoặc script từ nguồn ngoài.',
              '- Cả ba checkpoint nạp bằng **weights_only=True**. ConvNeXt strict state_dict; EfficientNet/YOLO dùng safe_globals chứa các lớp cụ thể đã kiểm tra. YOLO thay loader mặc định để ngăn weights_only=False và auto-install; dill._load_type được thay bằng hàm cục bộ chỉ chấp nhận set. Đây là giảm rủi ro pickle, không phải sandbox bảo mật.',
              '- Venv riêng `.venv-t024` kế thừa gói nền qua --system-site-packages để dùng PyTorch CUDA sẵn có; gói bổ sung cài trong venv, không sửa môi trường bài học Python 3.8. Venv không phải rào chắn bảo mật hệ điều hành.',
              '- timm 0.6.13 không có import paths timm.layers / timm.models._efficientnet_blocks được checkpoint lưu; chuyển sang **0.9.16**. YOLO names thực tế là logo/text; ánh xạ cả hai theo model card qfisch. EfficientNet có Linear(1280,1), sigmoid đúng một lần. ConvNeXt có head 768→512→256→2, softmax index 1.', '']
    for name, meta in results['models'].items():
        lines += [f'### {name}', '', f"Status: `{meta['status']}`; ảnh thành công: {meta.get('succeeded',0)}."]
        ck = meta.get('checkpoint')
        if ck and 'repo' in ck:
            lines += [f"- Repo/file: `{ck['repo']}/{ck['filename']}`; commit `{ck['revision']}`.",
                      f"- SHA-256: `{ck['sha256']}`; bytes={ck['bytes']}; download/cache={ck['download_or_cache_s']:.3f}s; load={meta.get('load_s',0):.3f}s."]
        elif ck:
            lines += [f"- Tổng checkpoint ba model: {ck['bytes']} bytes. {meta.get('timing_policy', '')}"]
        if 'adapter' in meta:
            lines += ['```json', json.dumps(meta['adapter'], ensure_ascii=False, indent=2), '```']
        lines += ['']
    lines += ['## Failure cases và sự cố', '',
              '- [Manifest ảnh lỗi](failure_cases/manifest.csv): tối đa 5 FP confidence cao nhất và 5 FN confidence thấp nhất mỗi model; preview thu nhỏ từ ảnh thật, giữ path và hash ảnh gốc. Nếu ít hơn 5 thì xuất toàn bộ lỗi hiện có.',
              f"- Lỗi chưa giải quyết của lần chạy này: **{len(results['errors'])}**. Xem [errors.json](errors.json); lịch sử kiểm tra tương thích ở inspection.log và compatibility_notes.md.", '',
              '## Tái lập', '', 'Từ `D:\\new`, PowerShell:', '', '```powershell',
              '.\\.venv-t024\\Scripts\\python.exe -m src.run_smoke_test --device auto --data-limit-mb 25 --seed 42 --threshold 0.5 --threads 4 --warmup 8 --output reports/latest',
              '.\\.venv-t024\\Scripts\\python.exe -m pytest -q tests', '```', '',
              'Revisions được giữ cố định khi chạy lại; input cached không ghi lại nếu không đổi. Xem [README project](../../README.md), requirements.txt và requirements-t024-lock.txt. Phiên bản/phần cứng đầy đủ nằm trong [results.json](results.json).', '',
              'Không commit, không push. Git được khởi tạo cục bộ ở branch feat/t024-watermark-smoke-test vì workspace ban đầu chưa có .git; data/checkpoints/venv bị ignore.', '',
              'Nguồn: [YOLO](https://huggingface.co/qfisch/yolov8n-watermark-detection), [EfficientNet](https://huggingface.co/qwertyforce/watermark_detection), [ConvNeXt code](https://github.com/boomb0om/watermark-detection), [weights](https://huggingface.co/boomb0om/watermark-detectors), [dataset](https://huggingface.co/datasets/boomb0om/watermarks-validation). YOLO AGPL-3.0 và tác giả ghi academic usage only; EfficientNet CC BY-SA 4.0; ConvNeXt HF repo BSD-3-Clause.']
    (output/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
