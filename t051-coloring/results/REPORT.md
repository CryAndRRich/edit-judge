# Kết quả T-051

## Dữ liệu và tiêu chí

5 ảnh đường phố có trong workspace được dùng với cùng một lệnh chuyển cảnh
sang hoàng hôn. Tra cứu URL Kaggle ngày 22/11/2026 trả HTTP 404. Phạm vi khảo
sát là 5 file có mã SHA-256 trong [sample.json](../sample.json).

- **SSIM độ sáng** đo mức tương đồng cấu trúc trên ảnh xám. Ảnh gốc được đổi kích
  thước theo ảnh đầu ra để tính. SSIM cũng nhạy với thay đổi ánh sáng hợp lệ.
- **Δ(R−B)** là thay đổi chênh lệch trung bình kênh đỏ và xanh lam, trên thang kênh
  8 bit. Giá trị dương cho thấy ảnh ngả đỏ hơn theo chỉ số này.
- **Δ độ sáng** dùng chênh lệch giá trị trung bình ảnh xám 8 bit.

Ảnh đối chiếu hiển thị cả ba hình cùng hàng để xem độ rõ của hiệu ứng hoàng hôn,
sự tự nhiên và mức giữ lại kiến trúc, người, vật thể. Dữ liệu hiện chỉ có 5 cảnh,
một lệnh và không có ảnh đích chuẩn.

## Đầu ra có sẵn

Hai ZIP trong `D:\DS` chứa đủ 5 PNG theo tên ảnh gốc cho mỗi model. Đây là dữ
liệu đầu ra từ workspace. ZIP chỉ chứa ảnh, nên lần chạy tạo ra chúng không có
log kèm theo để xác minh phiên bản checkpoint, thời gian hay VRAM.

| Model | Ảnh | SSIM độ sáng ↑ | Δ(R−B) | Δ độ sáng |
| --- | ---: | ---: | ---: | ---: |
| InstructPix2Pix | 5 | 0.3307 | +41.88 | −13.90 |
| MagicBrush | 5 | 0.2247 | +15.25 | −9.57 |

[Chỉ số theo ảnh](archived/comparison.csv) · [Trung bình theo model](archived/summary.csv)

- **InstructPix2Pix** có SSIM độ sáng cao hơn trên cả 5 ảnh. Ảnh `10048` giữ
  tương đối tốt mặt tiền và dây điện. Ảnh `10004` bị ám đỏ tím mạnh và mất
  nhiều chi tiết, còn `10073` xuất hiện quầng sáng ở mặt tiền.
- **MagicBrush** tạo nắng hoàng hôn đậm ở một số vị trí, đồng thời làm biến
  dạng kiến trúc hoặc thêm vùng cháy sáng. Hiện tượng thấy rõ ở `10004`,
  `10073` và `10082`.

| ZIP | SHA-256 |
| --- | --- |
| `instruct-pix2pix.zip` | `18a3344cf1df18b6ab8ddf9c4618fbf574700f82b05a7a9fe0e7a0f55af721c4` |
| `magicbrush.zip` | `3555096d66dceaec0783b3dfade4f8fdde922d2834ddc5c6e80e99d00ddf6f32` |

Ảnh đối chiếu cục bộ nằm tại `outputs/archived-comparison.png`. Từng ảnh
tách từ ZIP nằm trong `outputs/archived/<model>/`.

## Tái chạy trên GPU

RTX 4060 Laptop GPU nhận CUDA qua PyTorch 2.7.1+cu118. Đã kiểm tra mã chạy
và ghi log trên cả 5 ảnh bằng checkpoint tổng hợp nhỏ. Tải trọng số thật của
hai model bị gián đoạn. Các số liệu trong báo cáo lấy từ hai ZIP có sẵn.
