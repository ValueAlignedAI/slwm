import sys
from pathlib import Path

import pytest


def test_corrected_optimizer_generation_is_distinct_from_immutable_rejected_receipts(tmp_path,monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts/train"))
    from colab_archive_joint import variant_optimizer_prefix
    primary={"families":{"text":20,"audio":80}}
    legacy={"families":{"audio":80,"text":20}}
    config={**legacy,"training_family_order":["text","audio"]}
    lock=tmp_path/"training_lock.json"; lock.write_text("first-generation")
    assert variant_optimizer_prefix("depth6",legacy,lock,primary) is None
    first=variant_optimizer_prefix("depth6",config,lock,primary)
    assert first.startswith("g3-depth6-") and first!="g3-depth6"
    lock.write_text("changed-provenance")
    assert variant_optimizer_prefix("depth6",config,lock,primary)!=first
    assert variant_optimizer_prefix("structured",{**config,"families":{"text":21,"audio":79}},lock,primary) is None
