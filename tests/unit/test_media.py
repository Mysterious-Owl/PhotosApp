import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from facegroup.adapters.media.cache import DiskImageCache
from facegroup.adapters.media.decoder import PillowDecoder
from facegroup.adapters.media.files import (
    FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS,
    LocalFileSource,
    is_online_only,
)
from facegroup.ports.media import PermanentMediaError
from tests.conftest import make_photo


def test_walk_finds_photos_and_skips_hidden_system_and_other_files(tmp_path, unicode_name):
    make_photo(tmp_path / "a.jpg")
    make_photo(tmp_path / unicode_name / "b.PNG", fmt="PNG")
    make_photo(tmp_path / ".hidden" / "c.jpg")
    make_photo(tmp_path / "$RECYCLE.BIN" / "d.jpg")
    (tmp_path / "notes.txt").write_text("x")
    found = {f.rel_path for f in LocalFileSource().walk(tmp_path)}
    assert found == {"a.jpg", f"{unicode_name}/b.PNG"}


def test_online_only_attribute_detection():
    assert is_online_only(SimpleNamespace(st_file_attributes=FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS | 0x20))
    assert not is_online_only(SimpleNamespace(st_file_attributes=0x20))
    assert not is_online_only(SimpleNamespace())  # non-Windows stat


def test_decoder_reads_unicode_paths_and_caps_size(tmp_path, unicode_name):
    p = make_photo(tmp_path / unicode_name / "big.jpg", size=(4000, 3000))
    d = PillowDecoder(max_side=1000).decode(p)
    assert max(d.rgb.shape[:2]) == 1000
    assert (d.width, d.height) == (4000, 3000)
    assert len(d.content_hash) == 40


def test_decoder_applies_exif_rotation(tmp_path):
    p = make_photo(tmp_path / "rot.jpg", size=(400, 200), exif_orientation=6)  # rotate 90 on display
    d = PillowDecoder().decode(p)
    assert d.rgb.shape[:2] == (400, 200)  # now portrait: H=400, W=200
    assert (d.width, d.height) == (200, 400)


def test_decoder_reads_heic(tmp_path):
    p = make_photo(tmp_path / "x.heic", size=(320, 240), fmt="HEIF")
    d = PillowDecoder().decode(p)
    assert d.rgb.shape == (240, 320, 3)


def test_decoder_extracts_taken_at(tmp_path):
    p = tmp_path / "dated.jpg"
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = "2023:07:14 18:30:05"
    Image.new("RGB", (50, 50)).save(p, exif=exif.tobytes())
    assert PillowDecoder().decode(p).taken_at == "2023-07-14T18:30:05"


@pytest.mark.parametrize("content", [b"", b"not an image at all", b"\xff\xd8\xff\xe0" + b"\x00" * 50])
def test_corrupt_files_are_permanent_errors(tmp_path, content):
    p = tmp_path / "bad.jpg"
    p.write_bytes(content)
    with pytest.raises(PermanentMediaError):
        PillowDecoder().decode(p)


def test_missing_file_is_permanent(tmp_path):
    with pytest.raises(PermanentMediaError):
        PillowDecoder().decode(tmp_path / "gone.jpg")


def test_decompression_bomb_is_permanent(tmp_path):
    p = make_photo(tmp_path / "huge.png", size=(3000, 3000), fmt="PNG")
    with pytest.raises(PermanentMediaError):
        PillowDecoder(max_pixels=1_000_000).decode(p)


def test_cache_round_trip_and_cleanup(tmp_path):
    cache = DiskImageCache(tmp_path / "cache", thumbnail_size=64)
    img = np.zeros((300, 200, 3), np.uint8)
    img[:, :, 0] = 200
    cache.write_working(1234, img)
    back = cache.read_working(1234)
    assert back.shape == img.shape and abs(int(back[..., 0].mean()) - 200) <= 2
    cache.write_thumbnail(7, img)
    assert max(Image.open(cache.thumbnail_path(7)).size) == 64
    cache.delete_working(1234)
    assert not cache.working_path(1234).exists()
    with pytest.raises(FileNotFoundError):
        cache.read_working(1234)
    assert not list((tmp_path / "cache").rglob("*.tmp"))


def test_cache_works_under_unicode_folder(tmp_path, unicode_name):
    cache = DiskImageCache(tmp_path / unicode_name)
    cache.write_working(1, np.zeros((10, 10, 3), np.uint8))
    assert cache.read_working(1).shape == (10, 10, 3)
    assert os.path.exists(cache.working_path(1))
