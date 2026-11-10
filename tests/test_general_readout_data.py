import numpy as np
from training.general_readout_data import observed_robot_history, majority_activity


def test_current_and_future_commands_cannot_enter_actor_history():
    n=50; rng=np.random.default_rng(8)
    trajectory={"observed":rng.normal(size=(n,6)),"gripper":rng.random((n,1)),"commanded":rng.normal(size=(n,6)),"action":rng.normal(size=(n,7))}
    stats={"numeric_mean":np.zeros(64),"numeric_scale":np.ones(64),"action_mean":np.zeros(7),"action_scale":np.ones(7)}
    before=observed_robot_history(trajectory,3,32,stats)
    trajectory["commanded"][34:]+=100;trajectory["action"][34:]+=100
    after=observed_robot_history(trajectory,3,32,stats)
    assert all(np.array_equal(a,b) for a,b in zip(before,after))
    trajectory["commanded"][33]+=10
    assert not np.array_equal(before[0],observed_robot_history(trajectory,3,32,stats)[0])


def test_unlabelled_samples_do_not_become_a_fake_activity_class():
    assert majority_activity(np.array([0,0,2,1,2,1]))==1
    assert majority_activity(np.zeros(32,dtype=int))==0
