"""Deterministic local text overlays; original clean negatives avoid ambiguous labels."""
import csv
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from .download_data import sha256, write_json


def generate_synthetic(public, data_root, report_root, seed=42, limit_mb=25):
    root = Path(data_root) / 'synthetic'
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    clean = rng.sample(sorted((r for r in public if r['true_label'] == 0), key=lambda r: r['id']), 20)
    rows, encoded = [], []
    for i, source in enumerate(clean):
        with Image.open(source['path']) as im:
            base = ImageOps.exif_transpose(im).convert('RGB')
            base.thumbnail((960, 960))
        for variant in range(3):
            im = base.copy()
            text, opacity, angle, position, font_size = '', 0, 0, 'unchanged', 0
            if variant:
                text = ('SAMPLE', 'PREVIEW', 'WATERMARK')[(i + variant) % 3]
                w, h = im.size
                # Small corner text versus large central/diagonal text.
                font_size = max(10, int(min(w, h) * (0.045 if variant == 1 else 0.14)))
                font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', font_size)
                opacity = (45, 80, 120, 180, 225)[(i + variant) % 5]
                angle = 0 if variant == 1 else (-30, 0, 30)[i % 3]
                position = 'corner' if variant == 1 else ('diagonal' if angle else 'center')
                box = font.getbbox(text)
                tile = Image.new('RGBA', (box[2] - box[0] + 8, box[3] - box[1] + 8))
                # Alternating dark/light glyphs improve coverage on different backgrounds.
                color = (255, 255, 255) if i % 2 else (0, 0, 0)
                ImageDraw.Draw(tile).text((4-box[0], 4-box[1]), text, font=font, fill=(*color, opacity))
                if tile.width > w * .9:
                    tile.thumbnail((int(w*.9), h))
                tile = tile.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
                xy = (max(0, w-tile.width-8), max(0, h-tile.height-8)) if variant == 1 else ((w-tile.width)//2, (h-tile.height)//2)
                layer = Image.new('RGBA', im.size)
                layer.alpha_composite(tile, xy)
                im = Image.alpha_composite(im.convert('RGBA'), layer).convert('RGB')
            path = root / f'{i:02d}_{variant}.jpg'
            out = io.BytesIO()
            im.save(out, format='JPEG', quality=80, optimize=True)
            encoded.append((path, out.getvalue()))
            rows.append(dict(id=f'synthetic/{path.name}', dataset='synthetic', path=str(path.resolve()),
                             true_label=int(variant > 0), source_id=source['id'], source_sha256=source['sha256'],
                             text=text, opacity=opacity, angle=angle, position=position, font_size=font_size, seed=seed))
    total = sum(r['bytes'] for r in public) + sum(len(b) for _, b in encoded)
    if total > limit_mb * 1_000_000:
        return [], dict(status='skipped_budget', proposed_image_bytes=total)
    for row, (path, data) in zip(rows, encoded):
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
        row.update(sha256=sha256(path), bytes=len(data))
    with (Path(data_root) / 'synthetic_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(Path(data_root) / 'synthetic_manifest.json', rows)
    sheet = Image.new('RGB', (6*200, 10*165), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, row in enumerate(rows):
        with Image.open(row['path']) as im:
            im.thumbnail((196, 136)); x, y = (j%6)*200, (j//6)*165
            sheet.paste(im, (x, y))
            draw.text((x+2, y+137), f'{Path(row["path"]).stem} y={row["true_label"]} {row["text"]}', fill='black')
    sheet.save(Path(report_root) / 'synthetic_contact_sheet.jpg', quality=85)
    return rows, dict(status='ok', n=60, positive=40, negative=20, combined_image_bytes=total,
                      font='Arial Bold, C:/Windows/Fonts/arialbd.ttf', font_sha256=sha256('C:/Windows/Fonts/arialbd.ttf'),
                      negative_policy='Unchanged clean backgrounds, JPEG quality 80; no semantic overlays')
