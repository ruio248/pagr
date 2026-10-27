"""Record clean-venv smoke verification without publishing installation paths."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from pagr import runtime as rt


def main():
    root=Path(__file__).resolve().parents[1]
    cfg=(Path(sys.prefix)/'pyvenv.cfg').read_text().lower()
    assert 'include-system-site-packages = false' in cfg
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root,capture_output=True,text=True,check=True)
    subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,check=True)
    assert 'Ran 5 tests' in tests.stderr
    comparison=json.loads((root/'runs/clean_smoke/comparison.json').read_text())
    assert comparison['exact'] and comparison['method']=='am_adam'
    result={
        'date':'2026-10-27',
        'environment':{'python':platform.python_version(),'os':platform.system(),'arch':platform.machine(),'inherits_system_site_packages':False},
        'installed_distribution_versions':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())),
        'installation':'Non-editable project wheel and dependencies installed by pip into a newly created venv.',
        'pip_check':'passed','implementation_tests':{'passed':5,'failed':0},
        'integrity':rt.verify(),'full_baseline_replay':comparison,
        'notebook_executed_in_this_clean_environment':False,
        'hosted_ci':'Not verified: hosted runner did not start; this is a separate local macOS smoke test, not Ubuntu CI.'}
    (root/'provenance/clean_install_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('installed_distribution_versions','full_baseline_replay')},indent=2))


if __name__=='__main__':main()
