import csv
from PIL import Image
import pytest
from src.download_data import validate_manifest


def fixture_manifest(tmp_path, path='clean/a.png', label='False'):
    (tmp_path/'clean').mkdir(exist_ok=True)
    (tmp_path/'watermark').mkdir(exist_ok=True)
    Image.new('RGB',(4,4),'red').save(tmp_path/'clean/a.png')
    with (tmp_path/'validation_data.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','watermark']);w.writerow([path,label])


def test_valid_hash_and_rgb_read(tmp_path):
    fixture_manifest(tmp_path)
    r=validate_manifest(tmp_path,1,0)
    assert len(r[0]['sha256']) == 64 and r[0]['width'] == 4


@pytest.mark.parametrize('path,label', [('../outside.png','False'), ('clean/a.png','unknown'), ('clean/missing.png','False')])
def test_bad_manifest_fails_without_silent_skips(tmp_path,path,label):
    fixture_manifest(tmp_path,path,label)
    with pytest.raises((ValueError,FileNotFoundError)):
        validate_manifest(tmp_path,1,0)
