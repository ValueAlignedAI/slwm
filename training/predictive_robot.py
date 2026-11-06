"""G2 robot scale:1536 pinned shard records, verified identities and native clocks.

Only newly downloaded, reproducible record transports are discarded after cache
verification. Existing G1 sources/caches are never edited or removed here.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,re,struct
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from training import general_droid_sample as transport
from training.action_data_probe import array
from training.predictive_proto import example_type
from training.general_data import ROOT,CACHE,order,native_crc,checked_records,rgb,pose_delta
from training.general_robot_metadata import acquire as metadata,connected_roles
from training.general_sources import write
from training.research_sources import file_sha

FAMILY='droid_g2'
EXAMPLE=example_type()


def pin():
    root=ROOT/FAMILY;root.mkdir(parents=True,exist_ok=True);path=root/'source_plan.json'
    if path.exists():return json.loads(path.read_text())
    objects=json.loads((ROOT/'droid_expanded/objects.json').read_text())
    shards=sorted((v for v in objects if re.search(r'\.tfrecord-\d+-of-\d+$',v['name'])),key=lambda v:order(v['name']))
    assert len(shards)==2048
    result={'objects':shards[:1536],'episodes_requested':1536,'maximum_record_bytes':256*1024**2,'maximum_total_record_bytes':48*1024**3,'source_sha256':file_sha(__file__),'method':'First complete record of fixed hash-selected shards; nonuniform episode sample, record CRC/range/generation verified, no whole-shard checksum claim','raw_transport_retention':'New temporary downloads removed after decoded cache/receipt hashes; pinned public generation remains reproducible. G1 caches may be linked, never changed.'}
    write(path,result);return result


def native_clock(path,observed):
    """Validate one-to-one raw/RLDS poses before using raw robot timestamps."""
    import h5py
    with h5py.File(path,'r') as h:
        needed=('observation/robot_state/cartesian_position','observation/timestamp/robot_state/robot_timestamp_seconds','observation/timestamp/robot_state/robot_timestamp_nanos')
        if any(key not in h for key in needed):return None,{'verified':False,'reason':'Required raw pose/native clock datasets absent'}
        raw=np.asarray(h['observation/robot_state/cartesian_position'])
        # RLDS commonly omits the final raw frame. Accept only an exact full
        # prefix match with zero or one extra raw tail, never a nearest match.
        tail=len(raw)-len(observed)
        if tail not in (0,1) or raw.shape[1:]!=observed.shape[1:] or not np.allclose(raw[:len(observed)],observed,atol=2e-6,rtol=0):return None,{'verified':False,'reason':'Raw/RLDS pose alignment failed'}
        a=np.asarray(h['observation/timestamp/robot_state/robot_timestamp_seconds'],np.int64)
        b=np.asarray(h['observation/timestamp/robot_state/robot_timestamp_nanos'],np.int64)
        if len(a)!=len(raw) or len(b)!=len(raw):return None,{'verified':False,'reason':'Raw pose/clock length mismatch'}
        stamp=(a*1000000000+b)[:len(observed)];seconds=(stamp-stamp[0]).astype(np.float64)/1e9
        if not (np.isfinite(seconds).all() and np.all(np.diff(seconds)>0)):return None,{'verified':False,'reason':'Nonmonotonic/missing native robot clock'}
        return seconds,{'verified':True,'raw_pose_max_error':float(np.max(np.abs(raw[:len(observed)]-observed))),'raw_tail_frames_not_in_rlds':tail,'seconds':float(seconds[-1]),'median_interval_seconds':float(np.median(np.diff(seconds))) if len(seconds)>1 else None,'timestamp_source':'robot_state seconds+nanos, integer subtraction before float conversion; every RLDS pose matched to its raw prefix','raw_h5_sha256':file_sha(path)}


def episode(item,reuse,old_metadata):
    root=ROOT/FAMILY;receipt=root/'receipts'/(item['name'].rsplit('/',1)[-1]+'.json');receipt.parent.mkdir(exist_ok=True)
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved.get('complete'):
            if saved.get('row'):
                folder=CACHE/FAMILY/saved['row']['id']
                assert all(file_sha(folder/n)==h for n,h in saved['row']['files'].items())
            return saved
    previous=reuse.get(item['name']);card=None
    if previous:
        row=dict(previous);dest=CACHE/FAMILY/row['id'];dest.mkdir(parents=True,exist_ok=True)
        for name in ('trajectory.npz','rgb.npy'):
            target=dest/name
            if not target.exists():target.symlink_to((CACHE/'droid_expanded'/row['id']/name).resolve())
        info=old_metadata[row['id']];metadata_path=ROOT/'droid_expanded/raw_metadata'/row['id']
    else:
        card=transport.acquire(item,256*1024**2)
        payload=next(checked_records(card['file'],native_crc()))
        # Keep both the dynamic descriptor class and owning message alive.
        # The native protobuf map crashed when reached through temporaries.
        example=EXAMPLE.FromString(payload);features=example.features.feature
        n=len(features['steps/is_first'].int64_list.value);raw_path=features['episode_metadata/file_path'].bytes_list.value[0].decode()
        match=re.search(r'/([^/]+)/(?:success|failure)/(\d{4}-\d{2}-\d{2})/',raw_path);assert match,raw_path
        collector,date=match.groups();key='DROID-'+order(raw_path)[:20]
        row={'id':key,'raw_path':raw_path,'collector':collector,'session_date':date,'steps':n,'source_payload_sha256':card['sha256']}
        try:info=metadata(row,FAMILY)
        except ValueError as exc:
            if str(exc)!='Ambiguous metadata candidate':raise
            # An unresolved collector identity cannot enter any identity split.
            # Preserve every candidate listing, rather than choosing a user ID.
            info={'id':key,'metadata_unavailable':True,'reason':str(exc),'candidate_listings':{p.name:file_sha(p) for p in (root/'raw_metadata'/key).glob('*_listing.json')}}
        metadata_path=ROOT/FAMILY/'raw_metadata'/key
        if not info.get('metadata_unavailable'):
            dest=CACHE/FAMILY/key;dest.mkdir(parents=True,exist_ok=True)
            observed=array(features['steps/observation/cartesian_position'],6,n);commanded=array(features['steps/action_dict/cartesian_position'],6,n)
            gripper=array(features['steps/observation/gripper_position'],1,n);commanded_gripper=array(features['steps/action_dict/gripper_position'],1,n)
            assert all(np.isfinite(v).all() for v in (observed,commanded,gripper,commanded_gripper))
            state=np.concatenate([observed[:,:3],Rotation.from_euler('xyz',observed[:,3:]).as_matrix()[:,:,:2].reshape(n,6),gripper],1)
            action=np.concatenate([pose_delta(observed,commanded),commanded_gripper],1)
            np.savez(dest/'trajectory.npz',observed=observed.astype(np.float32),commanded=commanded.astype(np.float32),gripper=gripper.astype(np.float32),state=state.astype(np.float32),action=action.astype(np.float32))
            pixels=np.lib.format.open_memmap(dest/'rgb.npy',mode='w+',dtype=np.uint8,shape=(n,3,64,64,3))
            for view,name in enumerate(('wrist_image_left','exterior_image_1_left','exterior_image_2_left')):
                frames=features['steps/observation/'+name].bytes_list.value;assert len(frames)==n
                for tick,frame in enumerate(frames):pixels[tick,view]=rgb(frame)
            pixels.flush();del pixels
            row['captions']=sorted({v.decode() for name in ('language_instruction','language_instruction_2','language_instruction_3') for v in features['steps/'+name].bytes_list.value if v.decode().strip()})
    result={'complete':True,'object':item,'metadata':info,'row':None,'source_payload_sha256':row['source_payload_sha256']}
    if not info.get('metadata_unavailable'):
        observed=np.load(dest/'trajectory.npz')['observed'];clock=None;audit={'verified':False,'reason':'No bounded raw HDF5'}
        if (metadata_path/'trajectory.h5').exists():clock,audit=native_clock(metadata_path/'trajectory.h5',observed)
        if clock is None:clock=np.arange(len(observed),dtype=np.float64)
        np.save(dest/'clock.npy',clock);write(dest/'clock_audit.json',audit)
        row.update(physical_timestamps=audit['verified'],clock_audit=audit,files={p.name:file_sha(p) for p in dest.iterdir() if p.is_file()})
        result['row']=row
    write(receipt,result)
    if card:
        # Delete only this task's reconstructible transport, after durable receipt.
        assert Path(card['file']).is_relative_to(root/'sources')
        Path(card['file']).unlink()
    return result


def main():
    from google.protobuf.internal import api_implementation
    if api_implementation.Type()!='python':raise RuntimeError('Use PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python for isolated G2 acquisition')
    plan=pin();transport.FAMILY=FAMILY
    old_rows=json.loads((CACHE/'droid_expanded/rows.json').read_text());by_sha={r['source_payload_sha256']:r for r in old_rows}
    old_acquired=json.loads((ROOT/'droid_expanded/acquisition_complete.json').read_text())['verified']
    reuse={r['source_object']['name']:by_sha[r['sha256']] for r in old_acquired}
    old_metadata={r['id']:r for r in json.loads((ROOT/'droid_expanded/raw_metadata/results.json').read_text())}
    with ThreadPoolExecutor(max_workers=4) as pool:headers=list(pool.map(lambda item:transport.get_range(item,0,11),plan['objects']))
    sizes=[struct.unpack('<Q',h[:8])[0]+16 for h in headers];assert sum(sizes)<=plan['maximum_total_record_bytes'] and max(sizes)<=plan['maximum_record_bytes']+16
    root=ROOT/FAMILY;write(root/'record_budget.json',{'records':len(sizes),'bytes':sum(sizes)})
    results=[];errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(episode,item,reuse,old_metadata):item for item in plan['objects']}
        for future in as_completed(pending):
            try:results.append(future.result())
            except Exception as exc:errors.append({'object':pending[future],'error':repr(exc)})
            write(root/'acquisition_progress.json',{'completed':len(results),'decoded_verified':sum(r['row'] is not None for r in results),'errors':errors,'requested':len(pending)})
            if len(results)%16==0:print(json.dumps({'g2_robot_processed':len(results),'cached':sum(r['row'] is not None for r in results),'errors':len(errors)}),flush=True)
    assert not errors,'Keep failed sources; retry identical plan before fitting'
    rows=sorted((r['row'] for r in results if r['row']),key=lambda r:r['id']);assert len(rows)==len({r['id'] for r in rows})
    folder=root/'raw_metadata';folder.mkdir(exist_ok=True);write(folder/'results.json',[r['metadata'] for r in results])
    roles=connected_roles(rows,[r['metadata'] for r in results],FAMILY);assert roles['suitable_for_building_holdout']
    assert sum(r['role']=='train' for r in roles['rows'])>=400,'Insufficient verified robot training scale; do not silently launch'
    write(folder/'scene_role_plan.json',roles);write(CACHE/FAMILY/'rows.json',roles['rows'])
    write(root/'acquisition_complete.json',{'requested':len(plan['objects']),'verified_identity_episodes':len(rows),'physical_clock_episodes':sum(r['physical_timestamps'] for r in rows),'training_episodes':sum(r['role']=='train' for r in roles['rows']),'source_sha256':file_sha(__file__),'no_model_fit':True})


if __name__=='__main__':main()
