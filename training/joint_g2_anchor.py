"""Old frozen G2 endpoints on G3's new observations and few-shot supports."""
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from training.joint_sources import digest
from training.predictive_train import initialize

HASHES = {
    "G2-201": "e3c90e1cf5833cde93b536885bb571a7d856acf92707d8578e6ca952465f3448",
    "G2-202": "6c8602c3868b9307ada1701f52012a7180bc4782f6dcdb8c9f5c582f6d33a7fb",
    "G2-203": "750f4d54eb113225e7662145e9190248e73457946d111a96857d913e319ce391",
    "G2-204": "a3118c51940a2482b47fece585b14b838699f7df5a6e040409bb4c74ba0b7619"
}


class OldState(torch.nn.Module):
    def __init__(self,name):
        super().__init__()
        base = Path(os.environ.get("SLWM_G2_ENDPOINTS","experiments/multimodal/t3/predictive-bottleneck-20261106"))
        ck = base/name/"endpoint.pt"
        self.sha = digest(ck)
        if self.sha!=HASHES[name]:
            raise ValueError(name)
        saved = torch.load(ck,map_location="cpu",mmap=True,weights_only=False)
        self.model = initialize(saved["config"],saved["run"])
        self.model.load_state_dict(saved["model"])

    def representations(self,arrays):
        buffers = []
        for i in range(0,len(arrays["labels"]),64):
            s = slice(i,i+64)
            if "audio" in arrays:
                values = torch.from_numpy(arrays["audio"][s]).cuda()
                b,t = values.shape[:2]
                times = torch.arange(t,device="cuda")[None]*torch.tensor(arrays["duration"][s],device="cuda")[:,None]/t
                obs = {"audio":values[:,:,None].expand(-1,-1,4,-1),"audio_present":torch.ones(b,t,4,device="cuda",dtype=torch.bool)}
            elif "image" in arrays:
                values = torch.from_numpy(arrays["image"][s]).cuda()
                b,t = len(values),1
                rgb = values.reshape(b,4,4,3,8,8).permute(0,3,1,4,2,5).reshape(b,3,32,32)
                img = F.interpolate(rgb,size=(64,64),mode="bilinear",align_corners=False)
                obs = {"images":img[:,None,None],"image_present":torch.ones(b,1,1,device="cuda",dtype=torch.bool)}
                times = torch.zeros(b,t,device="cuda")
            else:
                text = torch.from_numpy(arrays["text"][s]).cuda()
                b,t = len(text),1
                length = torch.from_numpy(arrays["lengths"][s]).cuda()
                obs = {"text":text,"text_present":torch.arange(text.shape[1],device="cuda")[None]<length[:,None]}
                times = torch.zeros(b,t,device="cuda")
            obs.update(time=times,valid=torch.ones(b,t,device="cuda",dtype=torch.bool),time_unit=torch.zeros(b,device="cuda",dtype=torch.long))
            with torch.autocast("cuda",dtype=torch.bfloat16):
                value = self.model(obs)["shared"]
            buffers.append(value.float().cpu().numpy())
        return {"g2":np.concatenate(buffers)}
