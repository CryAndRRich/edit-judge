"""VOZ (voz.vn), a Vietnamese XenForo 2 forum.

VOZ has no request board, so discovery scans the listing pages of the forums where
people ask for photo fixes, and keeps threads whose title matches photo-editing keywords.
Guests see full-size attachments (lazy-loaded via data-src).

robots.txt sets `ai-train=no`: VOZ data may be analysed, but must not be used to train
or fine-tune models (README.md, "Data rules").
"""

import re

from .xenforo_site import XenForoSource

KEYWORDS = re.compile(
    r"(nhờ|giúp|help).{0,40}(ảnh|hình|photoshop|\bps\b)|"
    r"(chỉnh|sửa|phục\s*(hồi|chế)|làm\s*(nét|rõ)|xóa|xoá|tách|ghép|blend|retouch|restore)\s.{0,30}(ảnh|hình)|"
    r"photoshop|\bps\b ảnh|ảnh cũ|hình cũ",
    re.I,
)


class Voz(XenForoSource):
    name = "voz"
    label = "VOZ (Vietnam)"
    base_url = "https://voz.vn"
    listings = {
        "chup-anh-quay-phim": "/f/chup-anh-quay-phim.75/",
        "phan-mem": "/f/phan-mem.13/",
        "multimedia": "/f/multimedia.31/",
    }
    title_filter = KEYWORDS

    def probe_targets(self):
        return [("listing", self.base_url + self.listings["chup-anh-quay-phim"]),
                ("thread", self.base_url + "/t/nho-cac-dai-ca-giup-do-ve-chinh-anh-phuc-hoi-anh-cu-a.1043861/"),
                ("image", self.base_url + "/attachments/1000000121-jpg.2820155/")]
