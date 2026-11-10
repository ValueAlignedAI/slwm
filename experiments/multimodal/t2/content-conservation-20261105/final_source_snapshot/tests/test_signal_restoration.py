"""Hidden-evidence isolation and physical signal contracts for T2-R1."""
import numpy as np
import torch
from models.slwm_restoration import fill_observed,RestorationCodec,RestorationSLWM,restoration_loss,restoration_errors
from models.slwm_motion import MotionSLWM
from training.restoration_train import fixed_holes,crop
from training.restoration_data import flac_duration


def codec():
    old={"audio_mean":np.linspace(-5,2,64,dtype=np.float32),"audio_scale":np.ones(64,np.float32)*2,
         "audio_delta_mean":np.zeros(64,np.float32),"audio_delta_scale":np.ones(64,np.float32)}
    new={"mean":np.linspace(-3,1,64,dtype=np.float32),"scale":np.ones(64,np.float32)*3}
    return RestorationCodec(old,new)


def make(arm="signal"):
    torch.set_num_threads(1)
    cfg={"model":{"width":32,"layers":1,"heads":4,"packing":"aligned","processor":"signal"}}
    return RestorationSLWM(cfg,{"arm":arm,"seed":51}).eval()


def activate_head(model):
    with torch.no_grad(): model.head.output.weight.normal_(std=.01)


def test_interpolation_only_uses_visible_values_and_handles_empty_rows():
    x=torch.tensor([0.,999.,999.,3.,999.])[None,:,None]; seen=torch.tensor([[True,False,False,True,False]])
    torch.testing.assert_close(fill_observed(x,seen)[0,:,0],torch.tensor([0.,1.,2.,3.,3.]))
    torch.testing.assert_close(fill_observed(x,torch.zeros_like(seen)),torch.zeros_like(x))


def test_hidden_mel_and_its_deltas_cannot_change_inputs_or_predictions():
    c=codec(); y=torch.randn(2,32,256); seen=fixed_holes([{"sample_id":"a"},{"sample_id":"b"}],torch.device("cpu"))
    changed=y.clone(); changed[~seen]=torch.randn_like(changed[~seen])*1000
    before,after=c.observe(y,seen),c.observe(changed,seen)
    for key in ("audio","base","visible"): torch.testing.assert_close(before[key],after[key],atol=0,rtol=0)
    m=make(); activate_head(m); video=torch.randn(2,32,160); vm=torch.ones_like(seen)
    a=m(before["audio"],video,seen,vm,before["base"]); b=m(after["audio"],video,seen,vm,after["base"])
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert not torch.equal(before["target"],after["target"])


def test_unobserved_video_cannot_leak_and_one_window_has_correct_output_shape():
    m=make(); activate_head(m); c=codec(); seen=torch.ones(2,32,dtype=torch.bool); obs=c.observe(torch.randn(2,32,256),seen)
    video=torch.randn(2,32,160); absent=torch.zeros_like(seen)
    a=m(obs["audio"],video,seen,absent,obs["base"]); b=m(obs["audio"],video*1000,seen,absent,obs["base"])
    assert a.shape==(2,128,64); torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_channel_transform_preserves_actual_level_and_is_identity_at_zero():
    c=codec(); y=torch.randn(2,7,256)
    torch.testing.assert_close(c.augment(y,torch.zeros(2),torch.zeros(2,4)),y,atol=0,rtol=0)
    changed=c.augment(y,torch.tensor([10.,-10.]),torch.zeros(2,4))
    physical_delta=(changed-y).reshape(2,7,4,64)*c.scale100
    torch.testing.assert_close(physical_delta[0],torch.ones_like(physical_delta[0])*np.log(10),atol=3e-6,rtol=1e-5)


def test_signal_path_initialization_and_shared_decoder_initial_weights():
    signal,raw,vanilla=make(),make("raw_audio"),make("vanilla")
    for name,value in signal.head.state_dict().items():
        torch.testing.assert_close(value,raw.head.state_dict()[name],atol=0,rtol=0)
        torch.testing.assert_close(value,vanilla.head.state_dict()[name],atol=0,rtol=0)
    visible=torch.ones(1,32,dtype=torch.bool); visible[:,8:12]=False; obs=codec().observe(torch.randn(1,32,256),visible)
    result=signal(obs["audio"],torch.zeros(1,32,160),visible,torch.zeros_like(visible),obs["base"])
    torch.testing.assert_close(result,obs["base"],atol=0,rtol=0)


def test_objective_trains_processor_and_errors_respect_hidden_region():
    model=make(); activate_head(model); seen=torch.ones(2,32,dtype=torch.bool); seen[:,8:12]=False
    obs=codec().observe(torch.randn(2,32,256),seen); video=torch.randn(2,32,160)
    pred=model(obs["audio"],video,seen,torch.ones_like(seen),obs["base"])
    loss=restoration_loss(pred,obs["target"],seen); loss.backward()
    assert model.world.audio_adapter.project.weight.grad.abs().sum()>0
    assert model.head.output.weight.grad.abs().sum()>0
    perfect=restoration_errors(obs["target"],obs["target"],seen)
    assert all(value.shape==(2,) and value.count_nonzero()==0 for value in perfect.values())


def test_warmstart_removes_only_old_output_heads_and_frozen_core_stays_frozen():
    model=make("frozen_core"); old=MotionSLWM(model.world.cfg); model.initialize_world(old.state_dict())
    assert not any(p.requires_grad for p in model.world.parameters())
    torch.testing.assert_close(model.world.audio_adapter.project.weight,old.audio_adapter.project.weight)
    assert not hasattr(model.world,"audio_head") and not hasattr(model.world,"video_head")


def test_masks_and_crops_are_reproducible_and_do_not_extend_context():
    rows=[{"sample_id":"one"},{"sample_id":"two"}]; device=torch.device("cpu")
    mask=fixed_holes(rows,device); torch.testing.assert_close(mask.flip(0),fixed_holes(rows[::-1],device))
    assert torch.all(mask.sum(1)==24) and mask[:,0].all() and mask[:,-1].all()
    mel=torch.arange(2*80*256).reshape(2,80,256).float(); video=torch.zeros(2,80,160)
    y,v=crop({"mel100":mel,"video":video},np.array([1]),np.array([9]))
    assert y.shape==(1,32,256) and v.shape==(1,32,160); torch.testing.assert_close(y[0],mel[1,9:41])


def test_flac_streaminfo_duration_is_read_without_audio_decoder(tmp_path):
    data=bytearray(34); value=(16000<<44)|48000; data[10:18]=value.to_bytes(8,"big")
    path=tmp_path/"duration.flac"; path.write_bytes(b"fLaC"+bytes([128])+bytes([0,0,34])+data)
    assert flac_duration(path)==3.


def test_overlapping_output_windows_share_one_global_corruption_mask():
    from exploration.restoration_outputs import full_visibility,restore_sequence
    class Spy(torch.nn.Module):
        def __init__(self): super().__init__(); self.masks=[]
        def forward(self,a,v,seen,vm,base): self.masks.append(seen.clone()); return base
    length=70; seen=full_visibility({"sample_id":"global-mask"},length,torch.device("cpu"))
    obs={"audio":torch.randn(1,length,128),"visible":seen,"base":torch.randn(1,4*length,64)}
    model=Spy(); predicted=restore_sequence(model,obs,torch.zeros(1,length,160),False)
    np.testing.assert_allclose(predicted,obs["base"][0].numpy(),atol=1e-7)
    starts=[0,16,32,38]
    assert len(model.masks)==len(starts)
    for start,actual in zip(starts,model.masks): torch.testing.assert_close(actual,seen[:,start:start+32],atol=0,rtol=0)


def test_auxiliary_readout_leaves_primary_forward_unchanged():
    from models.slwm_content import ContentRestorationSLWM
    base=make(); activate_head(base)
    cfg={"model":{"width":32,"layers":1,"heads":4,"packing":"aligned","processor":"signal"}}
    auxiliary=ContentRestorationSLWM(cfg,{"arm":"signal","seed":51}).eval()
    missing=auxiliary.load_state_dict(base.state_dict(),strict=False)
    assert set(missing.missing_keys)=={"content_head.weight","content_head.bias"}
    seen=torch.ones(2,32,dtype=torch.bool); seen[:,4:8]=False; obs=codec().observe(torch.randn(2,32,256),seen); v=torch.randn(2,32,160)
    args=(obs["audio"],v,seen,torch.ones_like(seen),obs["base"])
    torch.testing.assert_close(base(*args),auxiliary(*args),atol=0,rtol=0)
    assert auxiliary.content_prediction is None


def test_auxiliary_gradient_reaches_world_without_a_primary_decoder_path():
    from models.slwm_content import ContentRestorationSLWM,content_loss
    cfg={"model":{"width":32,"layers":1,"heads":4,"packing":"aligned","processor":"signal"}}
    model=ContentRestorationSLWM(cfg,{"arm":"signal","seed":51}).train()
    with torch.no_grad(): model.content_head.weight.normal_(std=.01)
    seen=torch.ones(2,32,dtype=torch.bool); seen[:,4:8]=False; obs=codec().observe(torch.randn(2,32,256),seen)
    model(obs["audio"],torch.randn(2,32,160),seen,torch.ones_like(seen),obs["base"])
    loss=content_loss(model.content_prediction,obs["target"],seen)
    changed=obs["target"].clone().reshape(2,32,4,64); changed[~seen]=1000.
    torch.testing.assert_close(loss,content_loss(model.content_prediction,changed.reshape(2,128,64),seen),atol=0,rtol=0)
    loss.backward(); assert model.world.audio_adapter.project.weight.grad.abs().sum()>0
    assert model.head.output.weight.grad is None
