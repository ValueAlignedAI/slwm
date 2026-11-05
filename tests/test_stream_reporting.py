import pytest
pytest.importorskip("torch")
from training.broad_stream_report import interval


def test_person_interval_does_not_give_repeated_frames_extra_people_weight():
    simple=interval([0.,10.],["A","B"]); repeated=interval([0.]*20+[10.],["A"]*20+["B"])
    assert simple["person_mean"]==repeated["person_mean"]==5.
    assert simple["person_cluster_95_interval"]==repeated["person_cluster_95_interval"]==[0.,10.]
    assert repeated["people"]==2 and repeated["cases"]==21
