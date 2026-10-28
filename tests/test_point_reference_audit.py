"""Protocol and cache tests; no formal training or long audit is run here."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('point_reference_audit_test_module',
                                             ROOT/'scripts/audit_point_reference.py')
A = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = A
SPEC.loader.exec_module(A)


class ReferenceAuditProtocolTests(unittest.TestCase):
    def test_points_were_predeclared_and_cover_geometry(self):
        payload=json.loads(A.POINTS.read_text())
        points=payload['points']
        self.assertGreaterEqual(len(points),12)
        self.assertLessEqual(len(points),20)
        self.assertEqual(len({p['id'] for p in points}),len(points))
        old={(k,tuple(p),tuple(a)) for k,p,a,_ in A.M.fixed_reference_points()}
        c,_=A.M.load_config()
        for point in points:
            p=np.array([point['p_x'],point['p_y']])
            a=np.array([point['a_x'],point['a_y']])
            self.assertTrue(0 <= point['k'] < c.horizon)
            self.assertTrue(np.all((p >= 0)&(p <= 1)))
            self.assertGreater(np.linalg.norm(p-np.asarray(c.obstacle_center)),c.obstacle_radius)
            self.assertNotIn((point['k'],tuple(p),tuple(a)),old)
        names=' '.join(p['geometry_category'] for p in points)
        for needed in ('start','lower','upper','approach','after obstacle','goal',
                       'boundary','late horizon','early horizon','small','medium','saturated'):
            self.assertIn(needed,names)

    def test_nested_levels_change_one_axis_at_a_time(self):
        self.assertEqual(A.LEVELS[0][1:],(41,8,128))
        for earlier,later in zip(A.LEVELS,A.LEVELS[1:]):
            changed=sum(x!=y for x,y in zip(earlier[1:],later[1:]))
            self.assertEqual(changed,1,(earlier,later))
            self.assertTrue(all(y>=x for x,y in zip(earlier[1:],later[1:])))
        self.assertEqual(A.GRADIENT_SAMPLES,(2048,4096,8192))

    def test_cache_rejects_changed_actor(self):
        c=replace(A.M.Config(),horizon=2,flow_steps=2,warmup_steps=1)
        task=A.M.PointObstacle(c)
        actor=A.M.core.ConditionalVelocity(c)
        base=A.M.core.ConditionalVelocity(c)
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            level=('test',9,2,8)
            first=A.get_table(output,task,actor,base,level,A.POINTS.read_bytes())
            second=A.get_table(output,task,actor,base,level,A.POINTS.read_bytes())
            self.assertTrue(np.array_equal(first.values,second.values))
            with A.torch.no_grad():
                next(actor.parameters()).add_(1e-3)
            with self.assertRaises(RuntimeError):
                A.get_table(output,task,actor,base,level,A.POINTS.read_bytes())

    def test_formal_training_is_blocked_while_primary_gate_failed(self):
        c,_=A.M.load_config()
        summary=json.loads((ROOT/'runs/02-qam-point-obstacle-reference-audit-n0-v2'
                            /'reference-audit/precision_summary.json').read_text())
        self.assertFalse(summary['reference_validated'])
        with self.assertRaisesRegex(RuntimeError,'formal run blocked'):
            A.M.run_experiment(c,c.formal_seeds,'formal')


if __name__=='__main__':
    unittest.main()
