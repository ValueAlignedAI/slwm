"""Actual3090 BF16 backward/optimizer/serialization checks; no capability claim."""
import copy,fcntl,gc,json,time
from pathlib import Path
import torch

from models.predictive_world import PredictiveWorld
from training.general_batch import Sources,tensor_tree
from training.predictive_losses import loss
from training.general_sources import write
from training.research_sources import file_sha


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    cfg=json.loads(Path('configs/t3/predictive_world.json').read_text());old=json.loads(Path('configs/t2/general_transfer.json').read_text())
    # Old training data only, to validate implementation while expanded data is
    # acquired. These checks do not train or select any primary G2 checkpoint.
    source=Sources(old,'train');root=Path(cfg['root'])/'implementation_preflight_v2';root.mkdir(parents=True,exist_ok=True)
    records=[]
    with open('/tmp/slwm_research_gpu.lock','a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        for width,layers in [(512,10),(800,12)]:
            for variant in ('slwm','transformer'):
                model=PredictiveWorld(width=width,layers=layers,variant=variant,heads=8,slots=4,memory=512,phone_count=79,seed=777).cuda().train()
                teacher=copy.deepcopy(model).eval().requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,fused=True)
                for index,family in enumerate(cfg['family_schedule_100']):
                    size=cfg['batch_static'] if family=='coco' else cfg['batch_temporal']
                    batch=source.make(family,21_000_000+index,777,'cuda',size);times=[];objectives=[];torch.cuda.reset_peak_memory_stats()
                    for update in range(2):
                        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                        with torch.autocast('cuda',dtype=torch.bfloat16):objective,metrics=loss(model,teacher,batch,{},cfg,index)
                        assert torch.isfinite(objective);objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);optimizer.step();torch.cuda.synchronize()
                        times.append(time.perf_counter()-start);objectives.append(float(objective.detach()))
                    row={'width':width,'layers':layers,'variant':variant,'parameters':sum(p.numel() for p in model.parameters()),'family':family,'microbatch':size,'optimizer_steps':2,'seconds':times,'objective':objectives,'peak_bytes':torch.cuda.max_memory_allocated(),'history_gradient_nonzero':bool(model.history.weight_ih_l0.grad.abs().sum()>0),'final_people_used':0}
                    assert row['history_gradient_nonzero'];records.append(row);print(json.dumps(row),flush=True);write(root/'progress.json',records)
                    del batch,objective
                # Verify every full-size serialized parameter/buffer, not only count.
                path=root/(str(width)+'-'+variant+'.pt');torch.save({'model':model.cpu().state_dict(),'parameters':sum(p.numel() for p in model.parameters())},path)
                saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                assert saved['parameters']==sum(p.numel() for p in model.parameters())
                assert all(torch.equal(value,saved['model'][name]) for name,value in model.state_dict().items())
                del saved
                digest=file_sha(path);path.unlink();del model,teacher,optimizer;gc.collect();torch.cuda.empty_cache()
                write(root/(str(width)+'-'+variant+'-checkpoint.json'),{'serialized_sha256':digest,'temporary_weight_roundtrip_passed':True,'training_use':'None; implementation-only updates, no final predictions'})
    write(root/'complete.json',{'records':records,'source_sha256':file_sha(__file__),'primary_optimizer_steps':0,'actual_gpu_optimizer_checks':48,'final_observations_opened':False,'device':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda})


if __name__=='__main__':main()
