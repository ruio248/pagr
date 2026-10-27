import argparse
import json
from pathlib import Path
from . import runtime as rt


def main():
    p=argparse.ArgumentParser(description="PAGR research reproduction (not a validated new optimal-control solver)")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("verify",help="Verify all shipped sources and assets against SHA256")
    sub.add_parser("summary",help="Show all seven methods, including failures")
    for name in ("replay","replay-suite","prepare"):
        q=sub.add_parser(name)
        q.add_argument("--task",choices=rt.TASKS if name!="replay-suite" else (*rt.TASKS,"all"),required=name!="replay-suite",default="all")
        q.add_argument("--output",type=Path,required=True)
        if name!="replay-suite":q.add_argument("--seed",type=int,choices=rt.SEEDS,required=True)
        if name=="replay":q.add_argument("--method",choices=rt.METHODS,required=True)
        if name!="prepare":q.add_argument("--inputs",type=Path,help="Optional assets root produced by prepare")
        else:q.add_argument("--original-only",action="store_true",help="Skip RBF/residual/SiLU fitting")
    a=p.parse_args()
    if a.command=="verify":result=rt.verify()
    elif a.command=="summary":
        rt.verify()
        for task,methods in rt.summary()["tasks"].items():
            print(task)
            for method,row in methods.items():
                print(f"  {method:15s} cost={row['mean']:.6f} reduction={row['reduction_pct']:+.3f}% wins={row['wins']}/3")
        return
    elif a.command=="prepare":result=rt.prepare(a.task,a.seed,a.output,not a.original_only)
    elif a.command=="replay":result=rt.replay(a.task,a.seed,a.method,a.output,a.inputs)
    else:
        results=[]
        tasks=rt.TASKS if a.task=="all" else (a.task,)
        for task in tasks:
            for seed in rt.SEEDS:
                for method in rt.METHODS:
                    results.append(rt.replay(task,seed,method,a.output/task/f"seed_{seed}"/method,a.inputs))
        result={"runs":len(results),"max_final_metric_error":max(r["max_final_metric_error"] for r in results),
                "exact_runs":sum(r["exact"] for r in results),"comparisons":results}
        (a.output/"suite_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
