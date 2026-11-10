"""Before G2 data freeze, verify raw prefix alignment including one omitted tail."""
import json,shutil
from pathlib import Path
import numpy as np
from training.general_data import ROOT,CACHE
from training.general_sources import write
from training.predictive_robot import native_clock,FAMILY
from training.research_sources import file_sha


def apply_clock_revision(row,revised):
    """Keep G2 identities/roles; cached receipts can carry obsolete G1 labels."""
    assert row['id']==revised['id']
    return {**row,**{key:revised[key] for key in ('physical_timestamps','clock_audit','files')}}


def main():
    base=Path('artifacts/predictive_world/g2')
    root=ROOT/FAMILY;marker=root/'clock_revision.json'
    if marker.exists():assert json.loads(marker.read_text())['source_sha256']==file_sha(__file__);return
    assert not (base/'dataset_contract.json').exists(),'Clocks immutable after G2 data freeze'
    backup=base/'before_clock_prefix_revision';backup.mkdir(exist_ok=True)
    planpath=root/'raw_metadata/scene_role_plan.json';plan=json.loads(planpath.read_text());complete=root/'acquisition_complete.json'
    for source in (planpath,complete,CACHE/FAMILY/'rows.json'):
        target=backup/(source.parent.name+'_'+source.name)
        if not target.exists():shutil.copyfile(source,target)
    revised={};verified=0
    for receipt in sorted((root/'receipts').glob('*.json')):
        value=json.loads(receipt.read_text());row=value.get('row')
        if not row:continue
        folder=CACHE/FAMILY/row['id'];raw=root/'raw_metadata'/row['id']/'trajectory.h5'
        if not raw.exists():raw=ROOT/'droid_expanded/raw_metadata'/row['id']/'trajectory.h5'
        if raw.exists():
            observed=np.load(folder/'trajectory.npz')['observed'];clock,audit=native_clock(raw,observed)
            if clock is not None:
                np.save(folder/'clock.npy',clock);write(folder/'clock_audit.json',audit);row.update(physical_timestamps=True,clock_audit=audit)
                row['files'].update({'clock.npy':file_sha(folder/'clock.npy'),'clock_audit.json':file_sha(folder/'clock_audit.json')})
        revised[row['id']]=row;verified+=int(row['physical_timestamps']);write(receipt,value)
    plan['rows']=[apply_clock_revision(row,revised[row['id']]) for row in plan['rows']]
    write(planpath,plan);write(CACHE/FAMILY/'rows.json',plan['rows']);card=json.loads(complete.read_text());card.update(physical_clock_episodes=verified,clock_revision_source_sha256=file_sha(__file__));write(complete,card)
    write(marker,{'source_sha256':file_sha(__file__),'clock_function_sha256':file_sha('training/predictive_robot.py'),'verified_physical_clock_episodes':verified,'episodes':len(revised),'role_counts_unchanged':True,'rule':'Full raw prefix equal within2e-6; permit at most one omitted raw tail frame; strict native monotonic timestamps','primary_optimizer_updates':0})


if __name__=='__main__':main()
