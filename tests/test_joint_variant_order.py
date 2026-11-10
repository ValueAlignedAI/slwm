import json

import pytest

from training.joint_variant_config import read_variant_config


def test_sorted_metadata_preserves_registered_family_order(tmp_path):
    primary = {"families": {"text":20,"coco":25,"cifar":10,"speech":15,"ami":5,"clotho":5,"pamap2":10,"droid":10}}
    cfg = {**primary, "training_family_order":list(primary["families"]), "profile":"reduced"}
    path = tmp_path/"config.json"
    path.write_text(json.dumps(cfg,sort_keys=True))
    assert list(json.loads(path.read_text())["families"]) != cfg["training_family_order"]
    result = read_variant_config(path,primary)
    assert list(result["families"].items()) == list(primary["families"].items())
    assert result["profile"] == "reduced"


@pytest.mark.parametrize("update",[{}, {"training_family_order":["audio","text"]},
                                    {"training_family_order":["text","audio"],"families":{"text":51,"audio":49}}])
def test_rejects_missing_order_reordered_sources_or_changed_weights(tmp_path,update):
    primary = {"families":{"text":50,"audio":50}}
    path = tmp_path/"config.json"
    path.write_text(json.dumps({**primary,**update},sort_keys=True))
    with pytest.raises(ValueError):
        read_variant_config(path,primary)
