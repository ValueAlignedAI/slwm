from training.general_speech_score import aggregate
from training.general_speech_eval import paired_wer


def test_empty_owner_reference_keeps_insertions_separate():
    rows=[{"reference_words":10,"edits":1,"token_limit_reached":False},{"reference_words":0,"edits":30,"token_limit_reached":True}]
    score=aggregate(rows)
    assert score["wer"]==.1 and score["no_owner_reference_insertions"]==30
    assert score["token_limit_cases"]==1


def test_speech_gain_uses_independent_people_and_paired_word_counts():
    baseline=[{"id":str(i),"cluster":"person"+str(i),"reference_words":100,"edits":10} for i in range(5)]
    candidate=[{**r,"edits":8} for r in baseline]
    result=paired_wer(candidate,baseline)
    assert result["groups"]==5 and result["passes_half_point_gain"]
    assert result["gain"]==.02
