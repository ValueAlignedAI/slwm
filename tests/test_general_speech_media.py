import numpy as np
import torch
from models.slwm_general import GeneralWorld
from training.general_batch import base_observation
from training.general_speech_media import SpeechRenderer


class FakePhoneRenderer:
    def render_physical(self,phones,physical,present):return torch.zeros_like(physical)


def test_rejected_video_is_exactly_the_audio_only_path_even_for_fixed_fusion():
    renderer=SpeechRenderer.__new__(SpeechRenderer);renderer.device="cpu"
    renderer.stats={"audio_mean":np.zeros(100,np.float32),"audio_scale":np.ones(100,np.float32)};renderer.phone=FakePhoneRenderer()
    obs=base_observation(1,4,.04);obs.update(audio=np.ones((1,4,4,100),np.float32),audio_present=np.ones((1,4,4),bool))
    prepared={"obs":obs,"baseline":np.zeros((4,400),np.float32),"physical":np.ones((16,100),np.float32),"present":np.zeros(16,bool)}
    visual={"images":np.ones((1,4,1,3,64,64),np.float32),"image_present":np.ones((1,4,1),bool)}
    renderer.visual=lambda prepared,condition:None if condition=="audio" else visual
    model=GeneralWorld(variant="pooled_transformer",width=32,layers=1,heads=4,phone_count=79).eval()
    audio=renderer.infer(model,prepared,"audio")
    rejected=renderer.infer(model,prepared,"shift:8",threshold=2.)
    assert not rejected["video_accepted"]
    assert np.array_equal(audio["direct"],rejected["direct"])
    assert np.array_equal(audio["phones"],rejected["phones"])
    assert np.array_equal(audio["phone_probability"],rejected["phone_probability"])
