import pytest
pytest.importorskip("torch")
from training.broad_stream_report import interval,recognition_summary


def test_person_interval_does_not_give_repeated_frames_extra_people_weight():
    simple=interval([0.,10.],["A","B"]); repeated=interval([0.]*20+[10.],["A"]*20+["B"])
    assert simple["person_mean"]==repeated["person_mean"]==5.
    assert simple["person_cluster_95_interval"]==repeated["person_cluster_95_interval"]==[0.,10.]
    assert repeated["people"]==2 and repeated["cases"]==21


def test_empty_reference_insertions_remain_visible_separately_from_word_error_rate():
    rows=[{"wer":.5,"normalized_edit_count":2,"normalized_reference_words":4},
          {"wer":3.,"normalized_edit_count":3,"normalized_reference_words":0}]
    value=recognition_summary(rows)
    assert value["mean_clip_wer"]==1.75
    assert value["nonempty_reference_mean_clip_wer"]==.5
    assert value["nonempty_reference_corpus_wer"]==.5
    assert value["no_reference_cases"]==1 and value["no_reference_insertions"]==3
