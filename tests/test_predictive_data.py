"""Real expanded training-source integration; final observations remain closed."""
import json
from pathlib import Path
import numpy as np
import pytest
from training.predictive_data import Sources
from training.predictive_readout_data import ReadoutData


def test_expanded_robot_training_clocks_and_unseen_source_identity_contract():
    cfg=json.loads(Path('configs/t3/predictive_world.json').read_text())
    if not Path(cfg['dataset_manifest']).exists():pytest.skip('Expanded source integration runs on Fedora after preparation')
    source=Sources(cfg,'train')
    for position,context in [(0,32),(6,64)]:
        a=source.make('droid',position,777,'cpu',2);b=source.make('droid',position,777,'cpu',2)
        assert a['obs']['time'].shape==(2,context)
        assert np.allclose(a['obs']['time'],b['obs']['time']) and a['ids']==b['ids']
        assert (a['future_time'][:,0]>a['obs']['time'][:,-1]).all()
        assert a['planned_actions'].shape==(2,8,7)
        assert np.isfinite(a['obs']['numeric']).all()
    rows=json.loads(Path('artifacts/predictive_world/g2/har/rows.json').read_text());by_role={role:{r['person'] for r in rows if r['role']==role} for role in ('train','selection','test')}
    assert [len(by_role[r]) for r in ('train','selection','test')]==[16,5,9]
    assert not by_role['train']&by_role['selection'] and not by_role['train']&by_role['test']
    assert 'har' not in source.rows
    with pytest.raises(RuntimeError):ReadoutData(cfg,'har','test')
