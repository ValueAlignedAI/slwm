"""Integrate named, completed SLWM studies after testing the exact proposed GitHub main tree."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from g7_research_state import record
from g7_review_main import review

GH="ValueAlignedAI/slwm"
BRANCH="training/g7-continuation-alignment-control-20261110"
RESULT=Path("experiments/multimodal/t7/continuation-alignment-control-20261110")


def git(*args):
    env=dict(os.environ,GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/Users/db/dev/slwm-g2/artifacts/predictive_world/g2/github_known_hosts")
    return subprocess.check_output(["git",*args],text=True,env=env).strip()


def gh(*args):
    return subprocess.check_output(["gh",*args],text=True).strip()


def open_pr(branch,title,body):
    items=json.loads(gh("pr","list","--repo",GH,"--state","open","--base","main","--head",branch,"--json","number"))
    with tempfile.NamedTemporaryFile("w",suffix=".md") as f:
        f.write(body.rstrip()+"\n");f.flush()
        if items:
            number=items[0]["number"]
            gh("pr","edit",str(number),"--repo",GH,"--body-file",f.name)
        else:
            url=gh("pr","create","--repo",GH,"--base","main","--head",branch,"--title",title,"--body-file",f.name)
            number=int(url.rsplit("/",1)[1])
    return number


def integrate(repo,ctl,branch,head,title,body,final=False):
    for attempt in range(6):
        git("fetch","-q","origin","main",branch)
        remote_head=git("rev-parse","origin/"+branch)
        if remote_head!=head:raise ValueError("Published branch tip differs from the verified release candidate")
        base=git("rev-parse","origin/main");tree=git("merge-tree","--write-tree",base,head).splitlines()[0]
        verdict=review(tree)
        details=body+"\n\nFedora tested the exact proposed merge tree `"+tree+"` with pinned dependencies and real prepared fixtures. "+verdict["summary"]+"."
        num=open_pr(branch,title,details)
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            pr=json.loads(gh("pr","view",str(num),"--repo",GH,"--json","headRefOid,baseRefName,isDraft,mergeable,mergeStateStatus,statusCheckRollup,reviewDecision,state,url"))
            if pr["headRefOid"]!=head or pr["baseRefName"]!="main" or pr["isDraft"]:raise ValueError("Pull request no longer represents the reviewed study")
            if pr["state"]=="MERGED":break
            if pr["mergeable"]=="CONFLICTING" or pr["reviewDecision"]=="CHANGES_REQUESTED":raise RuntimeError("Pull request requires changes: "+pr["url"])
            if pr["mergeStateStatus"]=="CLEAN" and pr["mergeable"]=="MERGEABLE" and pr["reviewDecision"]!="REVIEW_REQUIRED":
                if gh("api","repos/"+GH+"/git/ref/heads/main","--jq",".object.sha")!=base:break
                gh("pr","merge",str(num),"--repo",GH,"--merge","--match-head-commit",head,"--subject",title+f" (#{num})")
                break
            time.sleep(8)
        else:raise TimeoutError("GitHub review/check requirements remain pending: "+pr["url"])
        merged=json.loads(gh("pr","view",str(num),"--repo",GH,"--json","state,mergeCommit,url,mergedAt"))
        if merged["state"]!="MERGED":continue
        git("fetch","-q","origin","main")
        subprocess.run(["git","merge-base","--is-ancestor",head,"origin/main"],check=True)
        receipt={"number":num,"url":merged["url"],"head":head,"main_sha":merged["mergeCommit"]["oid"],"tests_passed":verdict["passed"],"tested_tree":tree,"historical_or_held_controls":True,"merged_at":merged["mergedAt"]}
        record(repo,ctl,"Merged "+title+" after exact-tree review.",receipt,finished=final)
        return receipt
    raise RuntimeError("Main kept changing during exact-tree checks")


def finalized_readme():
    curves=json.loads((RESULT/"metrics/scale-curves.json").read_bytes())["models"]
    expense=json.loads((RESULT/"metrics/project-compute.json").read_bytes())
    if len([k for k,rs in curves.items() if k.startswith("g7-") and len(rs)==3 and rs[-1]["step"]==16384])!=16:
        raise ValueError("No completed 16-arm scale result")
    p=Path("README.md");s=p.read_text()
    start=s.index("### G7 shared state, speech and control")
    old="The registered scaling comparison continues both sizes and both architectures from their exact G6 optimizer states."
    if old not in s[start:] and "Sixteen exact continuations completed" in s[start:]:return False
    a=s.index(old,start);b=s.index("\n\n",a)
    def endpoints(c,arm):return " and ".join(f"{curves[f'g7-v5e-anchor-{c}-{seed}-{arm}'][-1]['fresh_test']['endpoint_px']:.2f}" for seed in (511,512))
    summary=f"Sixteen exact continuations completed at 16,384 updates on two named Colab v5e assignments, using {expense['project_compute_units']:.2f} project compute units. On 512 fresh Push-T starts, the 116.50M hybrid reached {endpoints('hybrid','fresh')} pixels with independent examples versus {endpoints('hybrid','replay')} on replay. The matched dense fresh-data models reached {endpoints('gpt2','fresh')} pixels. The release includes the larger model, equal-TPU-time analyses, held-room results and 43.69-second cross-modal examples. [Matched scaling results](experiments/multimodal/t7/continuation-alignment-control-20261110/report.md) · [Inference previews](experiments/multimodal/t7/continuation-alignment-control-20261110/examples/README.md)"
    p.write_text(s[:a]+summary+s[b:]);return True


def main(a):
    repo=a.repo.resolve();ctl=a.ctl.resolve();os.chdir(repo)
    if a.published:
        until=time.monotonic()+12*3600
        while time.monotonic()<until:
            record(repo,ctl)
            f=ctl/"publication.json"
            if f.exists():
                status=json.loads(f.read_bytes())
                if status.get("phase")=="published":break
            time.sleep(45)
        else:raise TimeoutError("The verified G7 publication has not completed")
        if git("branch","--show-current")!=BRANCH:raise ValueError("Research checkout changed")
        if git("rev-parse","HEAD")!=status["commit"]:raise ValueError("Active worktree differs from the published G7 result")
        if finalized_readme():
            git("diff","--check","--","README.md");git("commit","--only","-m","Update the README with completed scale, cost and inference results","--","README.md")
        git("push","origin","HEAD:refs/heads/"+BRANCH)
        receipt=integrate(repo,ctl,BRANCH,git("rev-parse","HEAD"),"Integrate verified G7 joint multimodal scaling and control",
            "Publishes all 16 completed continuations and their eight zero-update parents. The release includes 48 independently verified dual-disk generations, 80 fresh evaluations, dense Transformer controls, the clock-corrected fresh-speaker result, consecutive decisions, twelve continuous inference artifacts, project-only compute accounting and the README.",True)
    else:
        if not all((a.branch,a.head,a.title,a.body)):raise ValueError("Explicit branch, candidate SHA, title and PR body required")
        receipt=integrate(repo,ctl,a.branch,a.head,a.title,a.body.read_text())
    print(json.dumps(receipt,sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--ctl",type=Path,required=True)
    p.add_argument("--published",action="store_true");p.add_argument("--branch");p.add_argument("--head");p.add_argument("--title");p.add_argument("--body",type=Path)
    main(p.parse_args())
