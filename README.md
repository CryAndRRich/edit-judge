# edit-judge
EditJudge: Learning to Judge Photo Edits from the Crowd

## Watermark model — T-024

Tài liệu, mã nguồn và kết quả benchmark ba model phát hiện watermark nằm trong [experiments/T024](experiments/T024/README.md).

- [Kế hoạch benchmark](experiments/T024/T024_Watermark_Smoke_Test_Codex_Plan.md)
- [Báo cáo kết quả, so sánh model và giới hạn đánh giá](experiments/T024/reports/latest/README.md)
- [Bảng số liệu CSV](experiments/T024/reports/latest/summary.csv)
- [Kết quả kiểm tra tính nhất quán của artifacts](experiments/T024/reports/latest/verification.json)

Báo cáo gồm YOLOv8n, EfficientNetV2-B0, ConvNeXt-Tiny và majority voting trên 122 ảnh public + 60 ảnh synthetic. Dữ liệu gốc và checkpoint được tải theo revision đã pin khi chạy lại; không đưa vào Git.
