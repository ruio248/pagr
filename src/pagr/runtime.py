"""Portable paths and public entry points around the preserved training code."""
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
ASSETS = PACKAGE/"assets"
TASKS = ("navigation", "arm")
SEEDS = (18001, 18002, 18003)
METHODS = ("am_adam", "am_2x", "nominal_guard", "robust_guard", "smooth", "residual", "silu")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    manifest=json.loads((ASSETS/"manifest.json").read_text())
    failures=[]
    for name,digest in manifest.items():
        rel=Path(name).relative_to("src/pagr")
        path=PACKAGE/rel
        if not path.is_file() or sha(path)!=digest:
            failures.append(str(rel))
    if failures:
        raise RuntimeError("Bundled asset/source integrity mismatch: "+", ".join(failures))
    return {"verified_files":len(manifest), "status":"ok",
            "note":"Integrity is not algorithm efficacy. Historical solver warnings remain in the reference audit."}


def backend(asset_root=None):
    """Internal legacy module adapter. Call before any tensor deserialization."""
    verify()
    legacy=PACKAGE/"_legacy"
    for dirname in ("am_terminal_consistent_lean","am_finite_value_guard","am_data_anchored_gradient"):
        sys.path.insert(0,str(legacy/dirname))
    la=importlib.import_module("lean_am")
    aa=importlib.import_module("anchored_am")
    sn=importlib.import_module("smooth_network")
    vg=importlib.import_module("value_guard")
    assets=Path(asset_root).resolve() if asset_root else ASSETS
    la.PREVIOUS=assets/"original"
    aa.ROOT=sn.ROOT=assets/"anchored"
    return la,aa,sn,vg


def reference(task,seed,method):
    return json.loads((ASSETS/"reference/followup"/task/f"seed_{seed}"/method/"result.json").read_text())


def summary():
    return json.loads((ASSETS/"reference/summary.json").read_text())


def replay(task,seed,method,output,asset_root=None):
    if task not in TASKS or seed not in SEEDS or method not in METHODS:
        raise ValueError("Only the documented fixed tasks, seeds and methods are reference replays.")
    output=Path(output).resolve()
    if output==PACKAGE or PACKAGE in output.parents:
        raise ValueError("Write runs outside the installed package; bundled references are read-only.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {output}")
    la,aa,sn,vg=backend(asset_root)
    if method in METHODS[:4]:
        result=vg.train(task,seed,method,output)
    elif method in ("smooth","residual"):
        result=aa.train(task,seed,method,output)
    else:
        task_module,cfg,critics,c,_=sn.load(task,seed)
        result=la.train(task,cfg,seed,critics,c,la.Options(method="am_adam"),output)
    ref=reference(task,seed,method)
    errors={k:abs(float(v)-float(ref["final"][k])) for k,v in result["final"].items()}
    check={"task":task,"seed":seed,"method":method,"final":result["final"],
           "reference_final":ref["final"],"max_final_metric_error":max(errors.values()),
           "cost_error":errors["cost"],"exact":max(errors.values())==0.,
           "within_1e-4":max(errors.values())<=1e-4,
           "interpretation":"Reference replay, not a fresh-seed experiment. Cross-platform floating-point differences may occur."}
    (output/"comparison.json").write_text(json.dumps(check,indent=2)+"\n")
    return check


def prepare(task,seed,output,estimators=True):
    """Regenerate the original fixed-budget offline data/critics; no actor runs."""
    if task not in TASKS or seed not in SEEDS:
        raise ValueError("Use the documented tasks and seeds.")
    output=Path(output).resolve()
    if output==PACKAGE or PACKAGE in output.parents:
        raise ValueError("Generated inputs must be outside the installed package.")
    if (output/"original/pilot"/task/f"seed_{seed}").exists():
        raise FileExistsError("Refusing to overwrite a generated seed.")
    la,aa,sn,_=backend()
    task_module,cfg,_,_,_=la.load_inputs(task,seed)
    p=json.loads((ASSETS/"original/protocol.json").read_text())
    (output/"original").mkdir(parents=True,exist_ok=True)
    (output/"original/protocol.json").write_text(json.dumps(p,indent=2)+"\n")
    folder=output/"original/pilot"/task/f"seed_{seed}"
    la.rh.rel.task=task_module
    la.rh.rel.fit_ensemble(cfg,seed,la.rh.rel.Settings(members=5,updates=1000,batch=64),folder/"critics")
    manifest={p.name:sha(p) for p in sorted((folder/"critics").iterdir()) if p.is_file()}
    (folder/"input_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    la,aa,sn,_=backend(output)
    if estimators:
        aa.fit(task,seed,"smooth")
        aa.fit(task,seed,"residual")
        sn.fit(task,seed)
    return {"task":task,"seed":seed,"offline_states":cfg.critic_train+cfg.critic_valid,
            "offline_terminal_outcomes":(cfg.critic_train+cfg.critic_valid)*cfg.critic_mc,
            "estimators":"all" if estimators else "original ReLU only",
            "note":"Offline data generation is explicit. No new terminal observations are queried by actor replay."}
