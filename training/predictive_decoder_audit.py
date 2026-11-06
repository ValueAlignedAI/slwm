"""Independently reparse every newly decoded record preceding the native crash."""
import json
from pathlib import Path
import numpy as np
from google.protobuf.internal import api_implementation
from training import general_droid_sample as transport
from training.action_data_probe import array
from training.predictive_proto import example_type
from training.general_batch import rotation6d
from training.general_data import ROOT,CACHE,native_crc,checked_records,rgb,pose_delta
from training.general_sources import write
from training.research_sources import file_sha


def main():
    assert api_implementation.Type()=='python'
    transport.FAMILY='droid_g2';base=Path('artifacts/predictive_world/g2');revision=base/'acquisition_revision.json';cutoff=revision.stat().st_mtime
    old={r['id'] for r in json.loads((CACHE/'droid_expanded/rows.json').read_text())};Example=example_type();results=[]
    for receipt in sorted((ROOT/'droid_g2/receipts').glob('*.json')):
        if receipt.stat().st_mtime>cutoff:continue
        record=json.loads(receipt.read_text());row=record.get('row')
        if not row or row['id'] in old:continue
        card=transport.acquire(record['object'],256*1024**2);assert card['sha256']==row['source_payload_sha256']
        payload=next(checked_records(card['file'],native_crc()));message=Example.FromString(payload);features=message.features.feature;n=row['steps'];folder=CACHE/'droid_g2'/row['id'];cached=np.load(folder/'trajectory.npz')
        observed=array(features['steps/observation/cartesian_position'],6,n);commanded=array(features['steps/action_dict/cartesian_position'],6,n);gripper=array(features['steps/observation/gripper_position'],1,n);commanded_gripper=array(features['steps/action_dict/gripper_position'],1,n)
        expected={'observed':observed,'commanded':commanded,'gripper':gripper,'state':rotation6d(observed,gripper),'action':np.concatenate([pose_delta(observed,commanded),commanded_gripper],1)}
        checks={name:bool(np.array_equal(cached[name],value.astype(np.float32))) for name,value in expected.items()};assert all(checks.values()),checks
        pixels=np.load(folder/'rgb.npy',mmap_mode='r');count=0
        for view,name in enumerate(('wrist_image_left','exterior_image_1_left','exterior_image_2_left')):
            frames=features['steps/observation/'+name].bytes_list.value;assert len(frames)==n
            for tick,frame in enumerate(frames):assert np.array_equal(rgb(frame),pixels[tick,view]);count+=1
        results.append({'id':row['id'],'receipt_sha256':file_sha(receipt),'source_sha256':card['sha256'],'trajectory_arrays_equal':checks,'decoded_rgb_frames_equal':count})
        write(base/'decoder_recovery_audit_progress.json',results);Path(card['file']).unlink()
    write(base/'decoder_recovery_audit.json',{'passed':True,'records':results,'backend':'python','source_sha256':file_sha(__file__),'scope':'All novel v1 decoded records prior to native protobuf crash rechecked against complete CRC-verified public source payloads; G1 linked caches were already verified'})


if __name__=='__main__':main()
