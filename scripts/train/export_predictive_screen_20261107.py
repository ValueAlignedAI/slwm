"""Immutable inference-only exports from existing selected checkpoints; no fitting."""
import argparse,ast,fcntl,gc,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from training.general_sources import write
from training.research_sources import file_sha
from training.predictive_readout_data import ReadoutData
from training.predictive_features import supplied_text
from training.general_batch import tensor_tree
from training.general_consequences import images16
from exploration.predictive_infer import load_world,weight_hash
import tiktoken


def text_examples(canonical,dest):
 snapshot=dest/'source_snapshot';snapshot.mkdir(exist_ok=True)
 pending=[Path(name) for name in json.loads((canonical/'experiments/text/t1/500m-20261106/source_hashes.json').read_text()) if name.endswith('.py')]+[Path('exploration/scale_text_infer.py')];seen=set()
 while pending:
  name=pending.pop()
  if str(name) in seen:continue
  seen.add(str(name));source=canonical/name;target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
  for node in ast.walk(ast.parse(source.read_text())):
   modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.level==0 and node.module else []
   for module in modules:
    candidate=Path(*module.split('.')).with_suffix('.py')
    if (canonical/candidate).exists():pending.append(candidate)
 env=dict(os.environ,PYTHONPATH=str(canonical/'.research-venv/lib/python3.12/site-packages'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');samples=[]
 for folder in sorted(dest.glob('EXP-*')):
  output=folder/'sample.json'
  subprocess.run([sys.executable,'-m','exploration.scale_text_infer','--checkpoint',str((folder/'model.pt').resolve()),'--prompt','A robot observes a cup on the table. Before moving, it should','--tokens','64','--seed','81','--device','cpu','--output',str(output.resolve())],cwd=snapshot,env=env,check=True,stdout=subprocess.DEVNULL)
  samples.append(json.loads(output.read_text()))
 write(dest/'samples.json',samples)
 (dest/'README.md').write_text('# Interim text inference\n\nFour independent language checkpoints; exact training tokens, parameters and hashes are in manifest.json. These are not the shared multimodal weights. The500M final evaluation is pending. Optimizer/RNG resumes remain unchanged on Fedora.\n\nAll four fixed64-token CPU continuations are retained in samples.json, including poor ones. Loading and sampling passed from the included source snapshot without a corpus.\n\nFrom source_snapshot/:\n\n```bash\npython -m exploration.scale_text_infer --checkpoint ../EXP-T1-232/model.pt --prompt "A robot observes a cup on the table. Before moving, it should" --tokens 64 --seed 81 --device cpu\n```\n')


def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 cfg=json.loads(Path('configs/t3/predictive_world.json').read_text());root=Path(cfg['root']);out=root/'interim_inference_20261107';out.mkdir(exist_ok=True)
 assert not (root/'evaluation_lock.json').exists(),'This export uses selection examples only'
 (out/'source_snapshot').mkdir(exist_ok=True)
 for name in ('environment.json','source_hashes.json'):shutil.copyfile(root/name,out/name)
 for name,digest in json.loads((root/'source_hashes.json').read_text()).items():
  assert file_sha(name)==digest;target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
 data=ReadoutData(cfg,'droid','selection');batch=data.batch([0]);batch['obs'].update(supplied_text(batch['captions'],tiktoken.get_encoding('gpt2')))
 np.savez_compressed(out/'observations.npz',**batch['obs']);np.savez_compressed(out/'future.npz',time=batch['future_time'],plans=batch['plans'])
 frames,last,_=images16(data,batch,list(range(1,9)));np.savez_compressed(out/'reference_not_input.npz',images=frames,poses=batch['future_poses'])
 write(out/'case.json',{'id':batch['ids'][0],'group':batch['groups'][0],'role':'selection','selection_rule':'First fixed selection case; no output-quality selection','future_commands':'Supplied recorded command plan, not generated actions','final_tests_opened':False})
 cards=[];predictions={}
 with open('/tmp/slwm_research_gpu.lock','a') as lease:
  fcntl.flock(lease,fcntl.LOCK_EX)
  for run in cfg['runs'][:4]:
   folder=out/run['id'];folder.mkdir(exist_ok=True);source=root/run['id']/'best.pt';target=folder/'world.pt'
   if not target.exists():os.link(source,target)
   digest=file_sha(target);saved=torch.load(target,map_location='cpu',weights_only=False,mmap=True);assert saved['state']['position']==512;del saved
   command=[sys.executable,'-m','exploration.predictive_infer','--checkpoint',str(target.resolve()),'--observations',str((out/'observations.npz').resolve()),'--future',str((out/'future.npz').resolve()),'--output',str((folder/'inference.npz').resolve())]
   result=subprocess.run(command,cwd=out/'source_snapshot',check=True,capture_output=True,text=True);print(result.stdout,flush=True)
   model,saved=load_world(target);before=weight_hash(model);obs=tensor_tree(batch['obs'],'cuda');stats=saved['normalizers'];plan=torch.as_tensor((batch['plans']-stats['action_mean'])/stats['action_scale'],device='cuda',dtype=torch.float32)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    state=model(obs);again=model(obs);assert torch.equal(state['shared'],again['shared']);times=torch.from_numpy(batch['future_time']).cuda();views=[]
    for view in range(3):views.append(model.propose(model.query(state,times,obs['time_unit'],1,plan,view=view),'image').cpu().numpy())
   pixels=np.stack(views,2);predictions[run['id']]=pixels;np.savez_compressed(folder/'future_images.npz',values=pixels,times=batch['future_time'])
   cli=dict(np.load(folder/'inference.npz'));assert np.array_equal(cli['state'],state['shared'].cpu().numpy());assert all(np.isfinite(v).all() for v in cli.values()) and np.isfinite(pixels).all();assert 'command' not in cli
   after=weight_hash(model);assert before==after
   card={'run':run['id'],'variant':run['variant'],'seed':run['seed'],'parameters':sum(p.numel() for p in model.parameters()),'optimizer_updates':512,'checkpoint_sha256':digest,'checkpoint_bytes':target.stat().st_size,'source_commit':'36b473159da5e85714c63196b7eeab3901ff3ca3','state_shape':list(cli['state'].shape),'future_numeric_shape':list(cli['future_numeric'].shape),'future_images_shape':list(pixels.shape),'source_snapshot_cli_passed':True,'repeated_inference_equal':True,'weights_unchanged':True,'trained_actor_available':False,'no_final_test_observations':True}
   write(folder/'manifest.json',card);cards.append(card);del model,saved,state,again,obs,plan;gc.collect();torch.cuda.empty_cache()
 # A direct rendering of the first case, including poor predictions.
 scale=112;w=4*scale+40;h=3*(scale+28)+76;canvas=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(canvas)
 draw.text((10,5),'512-update models: fixed selection case / recorded future commands',fill='black')
 labels=['last observed','actual future +8','SLWM +8','Transformer +8']
 for c,label in enumerate(labels):draw.text((10+c*scale,28),label,fill='black')
 for view in range(3):
  arrays=[last[0,view],frames[0,-1,view],predictions['G2-201'][0,-1,view],predictions['G2-202'][0,-1,view]]
  for c,array in enumerate(arrays):
   pixels=np.clip((array.transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8);image=Image.fromarray(pixels).resize((scale-4,scale-4),Image.Resampling.NEAREST);canvas.paste(image,(10+c*scale,55+view*(scale+28)))
  draw.text((10,55+view*(scale+28)+scale),'view '+str(view),fill='black')
 canvas.save(out/'preview.png')
 write(out/'manifest.json',{'runs':cards,'role':'selection only','scope':'Interim state and recorded-command-conditioned forecasts; no learned actor, no final transfer result','unique_source_state':'Four selected512-update checkpoints, immutable hard links preserve bytes if best.pt is later replaced','created_unix':time.time(),'source_hashes_sha256':file_sha(root/'source_hashes.json')})
 (out/'README.md').write_text('''# Interim shared predictive-state inference\n\nFour124M-scale checkpoints, each trained for512 optimizer updates. These are early research artifacts, not completed12k runs or validated general-world/decision models. The fixed example is the first selection case; final test observations remain unopened. Predictions condition on supplied recorded future commands. No trained action head exists yet. The preview retains poor outputs.\n\n`manifest.json` records exact SHA256, parameter counts and real inference checks. Each folder contains `world.pt`, numeric/state outputs and three-view future RGB arrays. `observations.npz` and `future.npz` are the normalized inference inputs; targets are separate in `reference_not_input.npz`.\n\nFrom `source_snapshot/`, with recorded Torch/NumPy/SciPy/tiktoken dependencies:\n\n```bash\npython -m exploration.predictive_infer --checkpoint ../G2-201/world.pt --observations ../observations.npz --future ../future.npz --output /tmp/shared_prediction.npz --device cpu\n```\n\nThe core returns512D state and normalized numeric forecasts. Image forecasts have native16x16 RGB resolution. This is not free-running video generation. All four runs use the same example; no cases were selected for output quality.\n''')
 write(out/'export_complete.json',{'runs':len(cards),'actual_checkpoint_inference_passed':True,'new_optimizer_updates':0})
 # Strip optimizer state from the four interim text checkpoints; originals stay intact.
 text=Path('/home/db/dev/slwm/experiments/text/t1/500m-20261106');dest=text/'interim_inference_20261107';dest.mkdir(exist_ok=True);text_cards=[]
 for folder in sorted(text.glob('EXP-*')):
  source=folder/'best.pt'
  if not source.exists():continue
  saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True);target=dest/folder.name;target.mkdir(exist_ok=True)
  if (target/'model.pt').exists() and (target/'manifest.json').exists():
   text_cards.append(json.loads((target/'manifest.json').read_text()));del saved;gc.collect();continue
  keep={k:saved[k] for k in ('model','config','run','step','total_token_equivalent_step','provenance') if k in saved}
  with (target/'model.pt').open('wb') as stream:torch.save(keep,stream)
  card={'run':folder.name,'tokens_at_selected_checkpoint':saved['total_token_equivalent_step']*16384,'parameters':saved['provenance']['parameters'],'checkpoint_sha256':file_sha(target/'model.pt'),'source_checkpoint_sha256':file_sha(source),'checkpoint_bytes':(target/'model.pt').stat().st_size,'optimizer_removed':True,'original_resume_preserved':True,'scope':'Interim independent language weights, not shared multimodal knowledge or completed500M evaluation'};write(target/'manifest.json',card);text_cards.append(card);del saved,keep;gc.collect()
 write(dest/'manifest.json',{'runs':text_cards,'new_optimizer_updates':0});text_examples(Path('/home/db/dev/slwm'),dest);print(json.dumps({'world_export':str(out),'text_export':str(dest)}),flush=True)


if __name__=='__main__':main()
