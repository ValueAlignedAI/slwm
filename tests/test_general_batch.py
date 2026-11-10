"""Hidden-target isolation and predictive contracts on small synthetic fixtures."""
import copy
from pathlib import Path
import numpy as np
import torch
from training.general_batch import Sources,mask_support,numeric_baseline,rotation6d
from training.general_losses import pair_loss,batch_loss
from models.slwm_general import GeneralWorld


class FakeEncoder:
    eot_token=0
    def encode(self,text,**kwargs):return [3]*max(1,len(text.split()))


def source_fixture():
    source=Sources.__new__(Sources);source.cfg={"context":32,"horizon":8,"future_audio_guard_packets":2,"batch_temporal":2,"batch_static":2};source.encoder=FakeEncoder()
    source.stats={"audio_mean":np.zeros(100,np.float32),"audio_scale":np.ones(100,np.float32)}
    source.rows={"speech":[{"id":"utterance","actor":"person","group":"person","length":160,"offset":0,"mel_file":"mel","phone_file":"phones","spans":[(i*.1,(i+1)*.1,"word") for i in range(64)]}]}
    arrays={"mel":np.random.default_rng(5).normal(size=(160,400)).astype(np.float32),"phones":np.full((160,4),3,np.int64)};source.load=lambda name:arrays[str(name)]
    return source,arrays


def test_hidden_speech_targets_cannot_change_observations_or_baseline():
    source,arrays=source_fixture()
    for position in range(20):
        batch=source.make("speech",position,101,"cpu",1)
        if batch["modes"]==["repair"]:break
    start=int(batch["alignment_ids"][0].split(":")[-1]);present=batch["obs"]["audio_present"][0].numpy().reshape(-1)
    target=arrays["mel"][start:start+32].reshape(-1,100);target[~present]=10000
    changed=source.make("speech",position,101,"cpu",1)
    torch.testing.assert_close(batch["obs"]["audio"],changed["obs"]["audio"],rtol=0,atol=0)
    torch.testing.assert_close(batch["mel_baseline"],changed["mel_baseline"],rtol=0,atol=0)
    assert not torch.equal(batch["mel_target"],changed["mel_target"])
    assert torch.count_nonzero(batch["obs"]["audio"].masked_select(~batch["obs"]["audio_present"][...,None]))==0


def test_gap_support_matches_actual_fft_contamination():
    sound=torch.randn(32768);altered=sound.clone();altered[8192:12288]+=100
    window=torch.hann_window(1024)
    before=torch.stft(sound,1024,hop_length=256,window=window,return_complex=True)
    after=torch.stft(altered,1024,hop_length=256,window=window,return_complex=True)
    observed=mask_support(before.shape[-1],8192/24000,12288/24000)
    torch.testing.assert_close(before[:,observed],after[:,observed],rtol=0,atol=0)
    assert not torch.equal(before[:,~observed],after[:,~observed])


def test_numeric_interpolation_respects_irregular_physical_clock():
    values=np.array([[[0],[999],[999],[1]]],np.float32);present=np.array([[[True],[False],[False],[True]]]);time=np.array([[0,.01,.1,.11]])
    result=numeric_baseline(values,present,time)
    np.testing.assert_allclose(result[0,:,0],[0,1/11,10/11,1],rtol=1e-6)


def test_rotation_state_does_not_jump_at_euler_wrap():
    angles=np.array([[0,0,0,0,0,np.pi-.001],[0,0,0,0,0,-np.pi+.001]])
    encoded=rotation6d(angles,np.zeros((2,1)))
    assert encoded.shape==(2,10) and np.linalg.norm(encoded[1]-encoded[0])<.01


def test_multiptive_alignment_and_shuffled_control():
    values=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
    aligned=pair_loss(values,values,["a","a","b"],["c","c","d"])
    shuffled=pair_loss(values,values,["a","a","b"],["c","c","d"],shuffle=True)
    assert aligned<.001 and shuffled>1


def test_numeric_training_loss_has_finite_gradients_and_no_target_teacher_gradients():
    model=GeneralWorld(width=32,layers=1,heads=4,vocabulary=32,phone_count=11);teacher=copy.deepcopy(model).eval().requires_grad_(False)
    obs={"time":torch.arange(4)[None].float().expand(2,-1)*.02,"time_unit":torch.zeros(2,dtype=torch.long),"valid":torch.ones(2,4,dtype=torch.bool),"numeric":torch.randn(2,4,64),"numeric_present":torch.ones(2,4,64,dtype=torch.bool)}
    future={"time":torch.tensor([[.08,.10]]).expand(2,-1),"time_unit":obs["time_unit"],"valid":torch.ones(2,2,dtype=torch.bool),"numeric":torch.randn(2,2,64),"numeric_present":torch.ones(2,2,64,dtype=torch.bool)}
    batch={"family":"pamap2","obs":obs,"query_time":obs["time"],"numeric_target":obs["numeric"],"numeric_target_present":obs["numeric_present"],"numeric_hidden":~obs["numeric_present"],"numeric_baseline":obs["numeric"],"future_time":future["time"],"future_numeric":future["numeric"],"future_numeric_present":future["numeric_present"],"future_obs":future}
    cfg={"loss_weights":{name:1. for name in ("masked","future","latent","phones","ctc","alignment","correspondence","variance","risk")}}
    loss,metrics=batch_loss(model,teacher,batch,{},cfg,0);loss.backward()
    assert torch.isfinite(loss) and metrics["future"]>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    assert model.blocks[0].signal.real.grad is not None


def test_matched_processors_start_with_identical_common_weights():
    a=GeneralWorld("slwm",width=32,layers=2,heads=4,vocabulary=32,phone_count=11)
    b=GeneralWorld("transformer",width=32,layers=2,heads=4,vocabulary=32,phone_count=11)
    one=a.state_dict()
    for name,value in b.state_dict().items():torch.testing.assert_close(one[name],value,rtol=0,atol=0)
def test_short_records_never_enter_joint_window_sampling():
    from training.general_batch import Sources
    import numpy as np
    source=Sources.__new__(Sources);source.cfg={"context":32,"horizon":8,"future_audio_guard_packets":2};source.role="train"
    source.rows={"speech":[{"id":"short","group":"a","length":41},{"id":"exact","group":"b","length":42}]}
    assert {r["id"] for r in source.pick("speech",np.random.default_rng(1),20)}=={"exact"}


def test_sound_caption_sampling_gives_uploaders_equal_probability():
    source=Sources.__new__(Sources);source.cfg={};source.role="train"
    source.rows={"clotho":[{"id":str(i),"group":"many","steps":1000} for i in range(99)]+[{"id":"single","group":"one","steps":400}]}
    picked=source.pick("clotho",np.random.default_rng(1),4000)
    assert .45<sum(r["group"]=="one" for r in picked)/len(picked)<.55
