import numpy as np
from training.predictive_alignment import caption_scores
from training.predictive_dynamics_eval import controls,paired


def test_variable_caption_counts_and_duplicate_text_are_multi_positive():
    rows=[{'captions':['same','first']},{'captions':['same']},{'captions':['last','again','third']}]
    signal=np.eye(3,dtype=np.float32)
    captions=np.array([[0,1,0],[1,0,0],[1,0,0],[0,0,1],[0,0,1],[0,0,1]],np.float32)
    scores=caption_scores(signal,captions,rows)
    assert scores['records']==3 and scores['captions']==6
    assert scores['signal_to_text_recall1']==scores['text_to_signal_recall1']==1
    assert np.isclose(scores['uniform_signal_to_text_chance'],8/18)


def test_velocity_uses_measured_elapsed_time_and_bootstraps_groups():
    batch={'anchors':np.zeros((1,6)),'anchor_grippers':np.zeros(1),'commanded_poses':np.zeros((1,8,6)),'plans':np.zeros((1,8,7)),'previous_poses':np.array([[-1.,0,0,0,0,0,0]]),'obs':{'time':np.array([[0.,.2]])},'future_time':np.array([[.3,.5,.7,1.,1.2,1.5,1.7,2.]])}
    values,ratio,elapsed=controls(batch)
    assert np.allclose(ratio,[[.5,4.,9.]])
    assert np.allclose(values['last-velocity'][0,:,0],[.5,4.,9.])
    rows=[{'id':str(i),'group':str(i//2),'error':[1.,2.,3.]} for i in range(6)]
    baseline=[{**r,'error':[2.,4.,6.]} for r in rows]
    effect=paired(rows,baseline,'error')
    assert effect['groups']==3 and effect['gain_95ci']==[[1.,2.,3.],[1.,2.,3.]]
