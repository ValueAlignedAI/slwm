import os,subprocess,sys
import pytest


@pytest.mark.parametrize('backend',['python','upb'])
def test_dynamic_example_roundtrip_with_retained_owner_in_both_backends(backend):
    code="""from training.predictive_proto import example_type
T=example_type()
for i in range(20):
 x=T();x.features.feature['values'].float_list.value.extend([1.25,2.5]);x.features.feature['image'].bytes_list.value.append(b'raw-frame');x.features.feature['first'].int64_list.value.extend([1,0])
 parsed=T.FromString(x.SerializeToString());features=parsed.features.feature
 assert list(features['values'].float_list.value)==[1.25,2.5]
 assert features['image'].bytes_list.value[0]==b'raw-frame'
 assert list(features['first'].int64_list.value)==[1,0]
"""
    env=dict(os.environ,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=backend)
    subprocess.run([sys.executable,'-c',code],env=env,check=True,capture_output=True,text=True)
