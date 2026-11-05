import pytest
pytest.importorskip("torch")
pytest.importorskip("vocos")
pytest.importorskip("cmudict")
from training.broad_annotation_correction import restore


def test_unknown_spelling_recovery_preserves_word_times():
    spans,usable,_=restore({"transcript":"A PATRIARCHAL WILLOW","spans":[[0,.1,"a"],[.1,.7,"<unk>"],[.7,1.,"willow"]]})
    assert usable and spans==[(0,.1,"a"),(.1,.7,"patriarchal"),(.7,1.,"willow")]


def test_ambiguous_alignment_is_never_guessed():
    spans=[[0,.4,"<unk>"],[.4,.6,"'d"]]; actual,usable,_=restore({"transcript":"CONTINU'D","spans":spans})
    assert not usable and actual==spans
    _,usable,_=restore({"transcript":"CAT","spans":[[0,.2,"dog"]]}); assert not usable
