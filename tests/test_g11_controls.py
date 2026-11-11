"""Causal and measurement controls for the G11 instruction intervention."""
import io
from pathlib import Path
import tarfile
import sys
import numpy as np
import pytest

DATA=Path(__file__).resolve().parents[1]/"scripts/data"

def test_decoy_cannot_contain_a_later_or_identical_instruction(monkeypatch):
    monkeypatch.syspath_prepend(str(DATA))
    from g11_events import decoy
    def row(i,start,text,verb):
        return dict(id=i,start=start,end=start+.3,text=text,tokens=text.split(),content=text.split(),explicit=[verb],purpose="instruction")
    cur=row(3,8,"open cover","open")
    prior=row(0,3,"close lid","close")
    known=[row(1,4,"open cover","open"),prior,row(2,8.5,"rotate handle","rotate")]
    winner,conflict=decoy(cur,known)
    assert winner["id"]==0 and conflict
    assert decoy(cur,[known[0],known[2]])==(None,False)

def test_foreground_fixtures_and_no_change(monkeypatch):
    monkeypatch.syspath_prepend(str(DATA));monkeypatch.syspath_prepend(str(DATA.parent/"train"))
    from g11_measurement_audit import fixture
    r=fixture()
    assert r["static_view_change_fraction"]==0 and r["known_half_patch_shift_fraction"]==.5
    assert 23<r["recovered_mean_magnitude_mm"]<27 and r["hand_points_removed"]>0

def test_pax_depth_filename_with_truncated_regular_header(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(DATA))
    import g11_geometry as geo
    from PIL import Image
    blob=io.BytesIO();Image.fromarray(np.full((512,512),520,dtype=np.uint16)).save(blob,format="PNG");raw=blob.getvalue()
    tape=io.BytesIO()
    with tarfile.open(fileobj=tape,mode="w:",format=tarfile.PAX_FORMAT) as tar:
        name="member/"+"d"*102+"/000057.png";entry=tarfile.TarInfo(name);entry.size=len(raw);tar.addfile(entry,io.BytesIO(raw))
    data=tape.getvalue();monkeypatch.setattr(geo,"range_bytes",lambda key,off,n:data[off:off+n])
    frames=geo.DepthFrames((0,len(data)),tmp_path)
    np.testing.assert_array_equal(frames.get(57,60),np.full((512,512),520,dtype=np.uint16))
