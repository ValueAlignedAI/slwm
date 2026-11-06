"""Export usable state-only actors and report every frozen G2 result."""
import argparse,fcntl,gc,json,shutil
from pathlib import Path
import numpy as np
import tiktoken,torch

from training.general_sources import write
from training.predictive_features import supplied_text
from training.predictive_readout_data import ReadoutData
from training.predictive_train import atomic_save
from training.research_sources import file_sha
from exploration.predictive_infer import load_world,load_actor,actor_proposal
from training.general_batch import tensor_tree


def load(path):return json.loads(Path(path).read_text())


def export(cfg):
    root=Path(cfg['root']);assert (root/'evaluation_lock.json').exists();recipe=load('configs/t3/predictive_readouts.json');out=root/'exports';out.mkdir(exist_ok=True)
    for run in cfg['runs']:
        heads=[];folder=out/run['id'];folder.mkdir(exist_ok=True)
        for seed in recipe['seeds']:
            base=root/'readouts/heads/droid/actor'/run['id']/str(seed);complete=load(base/'complete.json');assert file_sha(base/'selected.pt')==complete['checkpoint_sha256'] and file_sha(base/'projection.npz')==complete['projection_sha256']
            saved=torch.load(base/'selected.pt',map_location='cpu',weights_only=False);fitted=dict(np.load(base/'projection.npz',allow_pickle=False))
            assert saved['inputs']==128 and saved['outputs']==7 and len(fitted['common_mean'])==0
            heads.append({'seed':seed,'model':saved['model'],'inputs':saved['inputs'],'outputs':saved['outputs'],'hidden':saved['hidden'],'projection':fitted})
        value={'run':run,'heads':heads,'core_endpoint_sha256':file_sha(root/run['id']/'endpoint.pt'),'feature':'shared512 only, float16 cache quantization then training-only128D SVD','physical_units':'body-frame translation meters, rotation vector radians, commanded gripper','status':'Internal proposals, no physical action or reward-trained policy'}
        atomic_save(folder/'actor.pt',value);write(folder/'manifest.json',{'core_endpoint':str(root/run['id']/'endpoint.pt'),'core_endpoint_sha256':value['core_endpoint_sha256'],'actor_sha256':file_sha(folder/'actor.pt'),'world_parameters':load(root/run['id']/'complete.json')['parameters'],'state_dim':512,'readout_seeds':recipe['seeds'],'raw_input_bypass':False})
    examples=out/'examples';examples.mkdir(exist_ok=True);encoder=tiktoken.get_encoding('gpt2')
    for family in ('har','droid'):
        data=ReadoutData(cfg,family,'test',True);batch=data.batch([0])
        if family=='droid':batch['obs'].update(supplied_text(batch['captions'],encoder))
        np.savez_compressed(examples/(family+'_observations.npz'),**batch['obs'])
        if family=='droid':np.savez_compressed(examples/'droid_future.npz',time=batch['future_time'],plans=batch['plans'])
        write(examples/(family+'_target.json'),{'id':batch['ids'][0],'group':batch['groups'][0],'target':int(batch['activity'][0]) if family=='har' else batch['target'][0].tolist(),'case_selection':'First frozen case, not chosen by model quality','observations_have_no_target_values':True})
    checks=[];values=dict(np.load(examples/'droid_observations.npz',allow_pickle=False));future=dict(np.load(examples/'droid_future.npz',allow_pickle=False))
    for run in cfg['runs']:
        with open('/tmp/slwm_research_gpu.lock','a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX);core=root/run['id']/'endpoint.pt';actor=out/run['id']/'actor.pt';model,saved=load_world(core);heads,policy=load_actor(actor);assert policy['core_endpoint_sha256']==file_sha(core)
            obs=tensor_tree(values,'cuda');stats=saved['normalizers'];plan=torch.as_tensor((future['plans']-stats['action_mean'])/stats['action_scale'],device='cuda',dtype=torch.float32)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                state=model(obs);again=model(obs);assert torch.equal(state['shared'],again['shared'])
                query=model.query(state,torch.from_numpy(future['time']).cuda(),obs['time_unit'],2,plan);prediction=model.propose(query,'numeric').cpu().numpy()
            command=actor_proposal(state,heads);assert np.isfinite(command).all() and np.isfinite(prediction).all()
            np.savez_compressed(out/run['id']/'fixed_example_prediction.npz',state=state['shared'].cpu().numpy(),command=command,future_numeric=prediction,future_time=future['time'])
            checks.append({'run':run['id'],'core_sha256':file_sha(core),'actor_sha256':file_sha(actor),'repeated_inference_equal':True,'proposal_shape':list(command.shape),'new_optimizer_updates':0})
            del model,heads,state,again,query,saved;gc.collect();torch.cuda.empty_cache()
    write(out/'actual_checkpoint_inference.json',checks)
    shutil.copyfile(cfg['normalizers'],out/'normalizers.npz');write(out/'config.json',cfg);write(out/'export_complete.json',{'actors':len(cfg['runs']),'actual_checkpoint_inference_checks':len(checks),'new_fit_updates':0,'final_lock_sha256':file_sha(root/'evaluation_lock.json')})


def report(cfg):
    root=Path(cfg['root']);transfer=load(root/'readouts/evaluation_summary.json');dynamics=load(root/'consequences/summary.json');alignment=load(root/'alignment/summary.json');speech=load(root/'speech_retention/summary.json');live=load(root/'operational/summary.json')
    completion={r['id']:load(root/r['id']/'complete.json') for r in cfg['runs']};dataset=load(cfg['dataset_manifest'])
    gates={'all_fixed_budgets_completed':all(r['target_reached'] for r in completion.values()),'har_superiority_both_slwm_seeds':transfer['transfer_gates']['har/activity'],'actor_superiority_both_slwm_seeds':transfer['transfer_gates']['droid/actor'],'har_raw_retention_both':all(r['within_margin_with_interval'] for r in transfer['tasks']['har/activity']['raw_retention'].values()),'actor_raw_retention_both':all(r['within_margin_with_interval'] for r in transfer['tasks']['droid/actor']['raw_retention'].values()),'history_dynamics_signal_seed201':dynamics['history_vs_point']['G2-201_vs_G2-211']['passes_five_percent_moving_dynamics'],'history_dynamics_transformer_seed201':dynamics['history_vs_point']['G2-202_vs_G2-212']['passes_five_percent_moving_dynamics'],'live_replay_completed':live['completed']}
    gates['registered_strong_transfer']=gates['all_fixed_budgets_completed'] and gates['har_superiority_both_slwm_seeds'] and gates['actor_superiority_both_slwm_seeds']
    cost={name:{'parameters':r['parameters'],'updates':r['optimizer_updates'],'position':r['position'],'selected_position':r['selected_position'],'active_training_hours':r['training_seconds']/3600,'validation_hours':r['validation_seconds']/3600,'target_reached':r['target_reached'],'microbatch_calls':r['family_calls']} for name,r in completion.items()}
    results={'gates':gates,'cost':cost,'dataset':dataset,'core_and_evaluation_sources_sha256':file_sha(root/'source_hashes.json'),'final_lock_sha256':file_sha(root/'evaluation_lock.json'),'scope':'Bounded predictive-state transfer, recorded action consequences and unactuated ongoing replay; not a general-world or physical-controller certification'}
    write(root/'findings.json',results);write(root/'gates.json',gates)
    lines=['# G2 predictive bottleneck and action-history findings','','All fixed endpoints and controls are retained. No final outcome selected a source role, checkpoint or readout.','','| Registered check | Outcome |','|---|---|']
    lines += ['| '+name+' | '+('PASS' if value else 'FAIL / insufficient evidence')+' |' for name,value in gates.items()]
    lines += ['','## Frozen unseen-task transfer','','| Task / run | Error | Strongest selection baseline | Paired gain95% interval |','|---|---:|---|---|']
    for task in ('har/activity','droid/actor'):
        item=transfer['tasks'][task]
        for name in [r['id'] for r in cfg['runs']]:
            effect=item['paired_effects'][name];lines.append(f"| {task} / {name} | {effect['candidate_error']:.4f} | {item['strongest_selection_baseline']} | {effect['gain_95ci']} |")
    lines += ['','HAR intervals resample its nine official final people; actor intervals resample connected buildings/collectors. Raw/edge actor controls retain17 explicit physical values within the same128-input head budget. The primary actor receives only compact state. Random cores, all input interventions, secondary aligned-state probes and MHealth remain in `readouts/evaluation_summary.json`. Activity error is one minus macro recall; actor error is training-scale-normalized absolute error, with physical metrics also retained.','','## Scale and cost','','| Run | Parameters | Updates | Training+validation h |','|---|---:|---:|---:|']
    for name,r in cost.items():lines.append(f"| {name} | {r['parameters']:,} | {r['updates']:,} | {r['active_training_hours']+r['validation_hours']:.2f} |")
    lines += ['','Both sizes use the same512-scalar state, source plans and per-seed effective example/packet budgets. Transformer references share the GRU posterior/dynamics so the primary pair isolates the field processor. Extra time is reported; no equal-compute claim or scaling law follows from two sizes. Bypass/point mechanism controls have one core seed each and require replication. The500M-token language models are independent weights; G2 text is supplied captions/goals.','','## Dynamics, content and ongoing inference','','The dynamics report retains1/4/8-step physical translation/SO(3)/gripper errors, normalized state MSE, image/background/motion errors, removed/shuffled plans and prefix shuffles with the final command fixed. Physical-second and ordinal-step velocities are separate. The registered mechanism gate is eighth-step moving physical-clock normalized pose MSE versus the matched point model, with5% relative improvement and a positive independent-group lower bound. Wrong plans have no observed counterfactual outcomes.','','`alignment/summary.json` measures independent caption retrieval and the fixed unpaired sound→image categories against random/edge controls. It is not matched-event recognition. `speech_retention/summary.json` measures nonsilent hidden-phone error on core-unexposed voices, without an external vocoder; phone error is not waveform intelligibility. The older G1 study independently evaluates speech intelligibility and long raw media.','','`operational/` logs persistent state, same-head reset controls, deadlines, stale rejection and exact weight hashes over more than40 seconds. Recorded observations do not respond to proposals. No physical device was actuated, and no reward/value function was learned. Data coverage failure remains explicit if no eligible native-time final episode exists.','','## Usable artifacts','','Each run retains `endpoint.pt`, validation-selected `best.pt`, a resumable optimizer/EMA checkpoint on Fedora, config/source/data hashes and full exposure logs. Inference snapshots sharing exactly the same bytes are hard-linked; no model state is approximated or compressed. `exports/<run>/actor.pt` contains the two frozen state-only heads and their training projections, tied to the endpoint SHA. Fixed HAR/robot observation/target examples are separate; no example is chosen by output quality.','','```bash','python -m exploration.predictive_infer --checkpoint experiments/multimodal/t3/predictive-bottleneck-20261106/G2-201/endpoint.pt --observations experiments/multimodal/t3/predictive-bottleneck-20261106/exports/examples/droid_observations.npz --actor experiments/multimodal/t3/predictive-bottleneck-20261106/exports/G2-201/actor.pt --output /tmp/world_proposal.npz','```','','Use the recorded code version/source snapshot. Observation examples are typed, normalized NPZ packets; raw physical inputs need the supplied normalizers and channel mapping. The `LiveWorld` API carries working memory and consumes only the latest encoded packet. Normal inference never changes weights.','','## Next decision','']
    if gates['registered_strong_transfer']:lines += ['The strong transfer gate passes on these sources/tasks. Next confirm on new data with longer memory and branching interactions before attaching goals/value learning or an actuator. This result alone does not establish general world understanding.']
    elif gates['har_raw_retention_both'] and gates['actor_raw_retention_both']:lines += ['The compact state retains competitive information, but the strong superiority gate does not pass. Preserve the compression/useful-task result, inspect action-history effects and cost, then test broader interacting environments. Do not relabel noninferiority as general-world superiority.']
    else:lines += ['The shared state does not meet the registered reuse/retention requirements. Retain task-specific gains and negative controls; inspect capacity, information loss, source imbalance and motion coverage before increasing parameters. A changed architecture needs fresh confirmation after these outcomes.']
    lines += ['','All data/source preparation failures, including the original protobuf decoder crash and the independent recovery audit, remain in the provenance record. Data counts and independent hours must remain distinct from repeated windows and camera tracks.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    write(root/'study_complete.json',{'findings_sha256':file_sha(root/'findings.json'),'report_sha256':file_sha(root/'REPORT.md'),'all_registered_phases_attempted':True,'gates':gates,'new_fit_after_final_lock':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('export','report'),required=True);a=p.parse_args();cfg=load('configs/t3/predictive_world.json');export(cfg) if a.stage=='export' else report(cfg)


if __name__=='__main__':main()
