"""Generate one bounded successor D68 run from verified D67 official D66 state.
All edits use frozen exact anchors and fail closed. Never launches itself, never
trains on audit games, and cannot create an unbounded recursive workflow.
"""
from __future__ import annotations
import argparse, json, re, pathlib, ast

ROOT=pathlib.Path(__file__).resolve().parents[1]
OLD="545dba627046621ec53b681d485978b9054776c717d894dcb0de55bec731d4a2"
OLD_INFER="fd5f832e0ded9519c23ad7469eee7a79f70d78f41bc0dab87c5f7bc57c8d478a"
CPP="c9340af265368547c37cbecc311d8d3982900885b433d1b4ccdc27fbdd7d041a"
OLD_STEP=10974
NEW_SEED="202611082320"
NEW_ARENA_SEED="202611082321"

def replace(s, before, after, count=None):
    n=s.count(before)
    if n==0 or (count is not None and n!=count):
        raise ValueError(f"source contract missing/ambiguous {n}/{count}: {before[:95]}")
    return s.replace(before,after)

def origin(path):
    p=ROOT/path
    if not p.is_file():raise ValueError("missing source "+path)
    return p.read_text()

def write(out,path,source):
    dest=out/path;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise ValueError("would overwrite D68 source "+path)
    dest.write_text(source)
    if path.endswith(".py"):ast.parse(source,filename=path)
    return path

def games_script(new, step, weight, training, selfplay, audit):
    s=origin("scripts/d66_games.py")
    s=replace(s,"from scripts import d66_native_game as native_game","from scripts import d68_native_game as native_game")
    s=replace(s,f'BASE="{OLD}"',f'BASE="{new}"')
    s=replace(s,"SEED=202611082020","SEED="+NEW_SEED)
    needle='SEARCH_SHA="'+CPP+'"'
    s=replace(s,needle,
        'D66_TRAIN_SHA="'+training+'"\n'
        'D66_SELFPLAY_SHA="'+selfplay+'"\n'
        'D66_AUDIT_SHA="'+audit+'"\n'+needle)
    s=replace(s,"d64_trained,d64_selfplay,d64_audit,out):",
                 "d64_trained,d64_selfplay,d64_audit,d66_trained,d66_selfplay,d66_audit,out):",1)
    s=replace(s,"(d64_trained,D64_TRAIN_SHA),(d64_selfplay,D64_SELFPLAY_SHA),(d64_audit,D64_AUDIT_SHA)):",
       "(d64_trained,D64_TRAIN_SHA),(d64_selfplay,D64_SELFPLAY_SHA),(d64_audit,D64_AUDIT_SHA),\n"
       "                     (d66_trained,D66_TRAIN_SHA),(d66_selfplay,D66_SELFPLAY_SHA),(d66_audit,D66_AUDIT_SHA)):")
    s=replace(s,'manifest["optimizer_step"]!=10974',f'manifest["optimizer_step"]!={step}',2)
    s=replace(s,f'manifest["weights_sha256"]!="{OLD_INFER}"',f'manifest["weights_sha256"]!="{weight}"')
    s=replace(s,'with tarfile.open(d64_trained,"r:gz") as t:','with tarfile.open(d66_trained,"r:gz") as t:')
    s=replace(s,'report["schema"]!="d64-learned1000-v1"','report["schema"]!="d66-learned1000-v1"')
    s=replace(s,'report["baseline"]!="e04e85dfebb507131df591bf0206e27771afe6944f3249124dfaf6eab5de51fb"',
      f'report["baseline"]!="{OLD}"')
    s=replace(s,'report["optimizer_step_before"]!=10080 or report["optimizer_step_after"]!=10974',
      f'report["optimizer_step_before"]!=10974 or report["optimizer_step_after"]!={step}')
    s=replace(s,'report["source_sha256"]!=D64_SELFPLAY_SHA','report["source_sha256"]!=D66_SELFPLAY_SHA')
    s=replace(s,'if opt is None or opt["step"]!=10974',f'if opt is None or opt["step"]!={step}')
    s=replace(s,'(d64_audit,"D64 audit",100,"d64-paired200-v1")):',
      '(d64_audit,"D64 audit",100,"d64-paired200-v1"),\n'
      '       (d66_selfplay,"D66 train",1000,"d66-d64-1000-v1"),\n'
      '       (d66_audit,"D66 audit",100,"d66-paired200-v1")):')
    s=replace(s,'plan={"schema":"d66-d64-1000-v1","baseline":BASE,"adam_step":10974',
      f'plan={{"schema":"d68-d66-1000-v1","baseline":BASE,"adam_step":{step}')
    s=replace(s,'"d64_current_training_sha256":D64_TRAIN_SHA,',
      '"d66_current_training_sha256":D66_TRAIN_SHA,\n'
      '          "d66_prior_training_games_sha256":D66_SELFPLAY_SHA,\n'
      '          "d66_prior_audit_sha256":D66_AUDIT_SHA,\n'
      '          "d64_historical_training_sha256":D64_TRAIN_SHA,')
    s=replace(s,'"input_trained_sha256":D64_TRAIN_SHA','"input_trained_sha256":D66_TRAIN_SHA')
    s=replace(s,'if opt["step"]!=10974',f'if opt["step"]!={step}')
    s=replace(s,'report={"schema":"d66-selfplay1000-result-v1"','report={"schema":"d68-selfplay1000-result-v1"')
    s=replace(s,"D66-D64-1000games.tar.gz","D68-D66-1000games.tar.gz")
    s=replace(s,'"d64-trained","d64-selfplay","d64-audit","out"',
       '"d64-trained","d64-selfplay","d64-audit","d66-trained","d66-selfplay","d66-audit","out"')
    s=replace(s,'x.d64_trained,x.d64_selfplay,x.d64_audit,x.out)',
       'x.d64_trained,x.d64_selfplay,x.d64_audit,x.d66_trained,x.d66_selfplay,x.d66_audit,x.out)')
    return s

def native_script(new, step):
    s=origin("scripts/d66_native_game.py")
    for a,b in (("D66_PARENT","D68_PARENT"),(OLD,new),
                ("202611082020",NEW_SEED),('opt["step"] != 10974',f'opt["step"] != {step}'),
                ("D66: v0.4.1 D64-adopted","D68: v0.4.1 D66-adopted")):
        s=replace(s,a,b)
    return s

def train_script(step):
    s=origin("scripts/d66_train.py")
    mapping=[
      ("from scripts import d66_games as games","from scripts import d68_games as games"),
      ("games.D64_TRAIN_SHA","games.D66_TRAIN_SHA"),
      ("games.D64_SELFPLAY_SHA","games.D66_SELFPLAY_SHA"),
      ('formal["optimizer_step"]!=10974',f'formal["optimizer_step"]!={step}'),
      ('report["schema"]!="d64-learned1000-v1"','report["schema"]!="d66-learned1000-v1"'),
      ('report["baseline"]!="e04e85dfebb507131df591bf0206e27771afe6944f3249124dfaf6eab5de51fb"',
       f'report["baseline"]!="{OLD}"'),
      ('report["optimizer_step_after"]!=10974',f'report["optimizer_step_after"]!={step}'),
      ('opt["step"]!=10974',f'opt["step"]!={step}'),
      ('plan["schema"]!="d66-d64-1000-v1"','plan["schema"]!="d68-d66-1000-v1"'),
      ('report["schema"]!="d66-selfplay1000-result-v1"','report["schema"]!="d68-selfplay1000-result-v1"'),
      ('old_plan["schema"]!="d64-d62-1000-v1"','old_plan["schema"]!="d66-d64-1000-v1"'),
      ('old_report["schema"]!="d64-selfplay1000-result-v1"','old_report["schema"]!="d66-selfplay1000-result-v1"'),
      ('if opt_after["step"]<=10974',f'if opt_after["step"]<={step}'),
      ('"optimizer_step_before":10974',f'"optimizer_step_before":{step}'),
      ('"prior_d64_replay_games":300','"prior_d66_replay_games":300'),
      ('"schema":"d66-learned1000-v1"','"schema":"d68-learned1000-v1"'),
      ("D66-D64-1000trained-unadopted.tar.gz","D68-D66-1000trained-unadopted.tar.gz"),
      ("D66: train current D64","D68: train current D66")]
    for a,b in mapping:s=replace(s,a,b)
    return s

def arena_script(step):
    s=origin("scripts/d66_arena.py")
    mapping=[
      ("from scripts import d66_games as games","from scripts import d68_games as games"),
      ("SEED=202611082021","SEED="+NEW_ARENA_SEED),
      ('opt["step"]!=10974 or after_opt["step"]<=10974',
       f'opt["step"]!={step} or after_opt["step"]<={step}'),
      ("games.D64_TRAIN_SHA","games.D66_TRAIN_SHA"),
      ('formal["optimizer_step"]!=10974',f'formal["optimizer_step"]!={step}'),
      ('parent_report["optimizer_step_after"]!=10974 or parent_report["schema"]!="d64-learned1000-v1"',
       f'parent_report["optimizer_step_after"]!={step} or parent_report["schema"]!="d66-learned1000-v1"'),
      ('report["schema"]!="d66-learned1000-v1"','report["schema"]!="d68-learned1000-v1"'),
      ('report["optimizer_step_before"]!=10974 or report["optimizer_step_after"]<=10974',
       f'report["optimizer_step_before"]!={step} or report["optimizer_step_after"]<={step}'),
      ('train_plan["schema"]!="d66-d64-1000-v1"','train_plan["schema"]!="d68-d66-1000-v1"'),
      ('(d64_audit,"D64 audit",100,"d64-paired200-v1")):',
       '(d64_audit,"D64 audit",100,"d64-paired200-v1"),\n'
       '        (d66_selfplay,"D66 train",1000,"d66-d64-1000-v1"),\n'
       '        (d66_audit,"D66 audit",100,"d66-paired200-v1")):'),
      ('d64_selfplay,d64_audit,out):','d64_selfplay,d64_audit,d66_selfplay,d66_audit,out):'),
      ('(d64_selfplay,games.D64_SELFPLAY_SHA),(d64_audit,games.D64_AUDIT_SHA)):',
       '(d64_selfplay,games.D64_SELFPLAY_SHA),(d64_audit,games.D64_AUDIT_SHA),\n'
       '                     (d66_selfplay,games.D66_SELFPLAY_SHA),(d66_audit,games.D66_AUDIT_SHA)):'),
      ('"schema":"d66-paired200-v1"','"schema":"d68-paired200-v1"'),
      ('plan["schema"]!="d66-paired200-v1"','plan["schema"]!="d68-paired200-v1"'),
      ('"schema":"d66-paired200-result-v1"','"schema":"d68-paired200-result-v1"'),
      ('"inconclusive_keep_D64"','"inconclusive_keep_D66"'),
      ("D66-D64-before-after-200games.tar.gz","D68-D66-before-after-200games.tar.gz"),
      ('"d64-selfplay","d64-audit","out"',
       '"d64-selfplay","d64-audit","d66-selfplay","d66-audit","out"'),
      ('z.d64_selfplay,z.d64_audit,z.out)',
       'z.d64_selfplay,z.d64_audit,z.d66_selfplay,z.d66_audit,z.out)'),
      ("D66: fresh round","D68: fresh round")]
    for a,b in mapping:s=replace(s,a,b)
    return s


def workflow_script(new,step,weight,run,training,selfplay,audit):
    s=origin(".github/workflows/d66-d64-v041-1000-selfplay-train-audit.yml")
    s=replace(s,"D66","D68")
    s=replace(s,"d66","d68")
    s=replace(s,"D68-D64","D68-D66")
    s=replace(s,"d68-d64","d68-d66")
    s=replace(s,"D68 v0.4.1 D64 PVS+mate","D68 v0.4.1 D66 PVS+mate")
    s=replace(s,"d68-20261108-d64-v041-selfplay1000-20runner",
                  "d68-20261108-d66-v041-selfplay1000-20runner")
    s=replace(s,OLD,new)
    s=replace(s,OLD_INFER,weight)
    s=replace(s,str(OLD_STEP),str(step))
    s=s.replace("group: fukaji-d68-v041-d64-1000-selfplay-audit",
                "group: fukaji-d68-v041-d66-1000-selfplay-audit")
    downloads=(
        f"          gh release download fukaji-d66-d64-trained-{run} --pattern D66-D64-1000trained-unadopted.tar.gz --dir input\n"
        f"          gh release download fukaji-d66-d64-1000-selfplay-{run} --pattern D66-D64-1000games.tar.gz --dir input\n"
        f"          gh release download fukaji-d66-d64-before-after-200games-{run} --pattern D66-D64-before-after-200games.tar.gz --dir input")
    checks=(
        f"          echo '{training}  input/D66-D64-1000trained-unadopted.tar.gz' | sha256sum -c -\n"
        f"          echo '{selfplay}  input/D66-D64-1000games.tar.gz' | sha256sum -c -\n"
        f"          echo '{audit}  input/D66-D64-before-after-200games.tar.gz' | sha256sum -c -")
    old_download="          gh release download fukaji-d64-d62-before-after-200games-42465412656 --pattern D64-D62-before-after-200games.tar.gz --dir input"
    old_sha="          echo '3a4b96fcf7d56975124704b6296734c912ff065dad2dbdb853004d051e7cb88e  input/D64-D62-before-after-200games.tar.gz' | sha256sum -c -"
    s=replace(s,old_download,old_download+"\n"+downloads,2)
    s=replace(s,old_sha,old_sha+"\n"+checks,2)
    i=s.index("  wave0:\n");j=s.index("  train:\n");k=s.index("  audit_prepare:\n");h=s.index("  audit:\n",k)
    if not 0<i<j<k<h:raise ValueError("D68 run job boundary missing")
    pre,waves,t,ap,post=s[:i],s[i:j],s[j:k],s[k:h],s[h:]
    old_flag="--d64-audit input/D64-D62-before-after-200games.tar.gz --out prepared"
    pre=replace(pre,old_flag,
       "--d64-audit input/D64-D62-before-after-200games.tar.gz "
       "--d66-trained input/D66-D64-1000trained-unadopted.tar.gz "
       "--d66-selfplay input/D66-D64-1000games.tar.gz "
       "--d66-audit input/D66-D64-before-after-200games.tar.gz --out prepared")
    ap=replace(ap,old_flag,
       "--d64-audit input/D64-D62-before-after-200games.tar.gz "
       "--d66-selfplay input/D66-D64-1000games.tar.gz "
       "--d66-audit input/D66-D64-before-after-200games.tar.gz --out prepared")
    # The train job restores the immediate parent, not the two-generation-old D64.
    for before,after in [
       ("fukaji-d64-d62-trained-42465412656",f"fukaji-d66-d64-trained-{run}"),
       ("fukaji-d64-d62-1000-selfplay-42465412656",f"fukaji-d66-d64-1000-selfplay-{run}"),
       ("D64-D62-1000trained-unadopted.tar.gz","D66-D64-1000trained-unadopted.tar.gz"),
       ("D64-D62-1000games.tar.gz","D66-D64-1000games.tar.gz"),
       ("ae93fbe93c2de9dde16ba84945e13aca62f1c122a5edb99cd3796795f6fce2ea",training),
       ("5def465bc811824d380001f40360055171b26e2c3202710717d1b2509391e56d",selfplay)]:
        t=replace(t,before,after)
    ap=replace(ap,"--baseline-archive input/D64-D62-1000trained-unadopted.tar.gz",
                 "--baseline-archive input/D66-D64-1000trained-unadopted.tar.gz")
    pre=replace(pre,'parent_report["schema"]!="d64-learned1000-v1"',
                    'parent_report["schema"]!="d66-learned1000-v1"')
    pre=replace(pre,"inconclusive_keep_D64","inconclusive_keep_D66")
    need=['report["schema"]!="d68-learned1000-v1"',
          'plan={"schema":"d68-paired200-v1"',
          'if plan["schema"]!="d68-paired200-v1"']
    if any(tkn not in pre for tkn in need):
        raise ValueError("D68 learned/producer/consumer schema guard missing")
    s=pre+waves+t+ap+post
    mandatory=["research/d68/START.json","scripts/d68_games.py","scripts/d68_train.py",
              "scripts/d68_arena.py","D68-D66-1000games.tar.gz",
              "D68-D66-1000trained-unadopted.tar.gz",
              "D68-D66-before-after-200games.tar.gz",
              f"fukaji-d66-d64-trained-{run}", "d68-r0-lane-",
              f"'weights_sha256':'{weight}'","parent_report[\"schema\"]!=\"d66-learned1000-v1\""]
    missing=[t for t in mandatory if t not in s]
    if missing:raise ValueError("generated D68 workflow missing "+repr(missing))
    return s

def generate(out, *,new,step,weight,run,training,selfplay,audit,preview=False):
    for key,value in (("new",new),("weight",weight),("training",training),
                      ("selfplay",selfplay),("audit",audit)):
        if len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid SHA "+key)
    if type(step) is not int or not 10974<step<=20000 or type(run) is not int or run<=0:
        raise ValueError("invalid D66 run or full Adam step")
    if not preview:
        m=json.loads((ROOT/"formal/model.json").read_text())
        expect={"research_snapshot_sha256":new,"weights_sha256":weight,
                "optimizer_step":step,"search_backend_variant":"d48_pvs_mate",
                "search_implementation_sha256":CPP,"version":"FUKAJI v0.4.1"}
        for k,v in expect.items():
            if m.get(k)!=v:raise ValueError("D67 formal model drift "+k)
    if out.resolve()==ROOT.resolve():
        for name in ("scripts/d68_native_game.py","scripts/d68_games.py",
                     "scripts/d68_train.py","scripts/d68_arena.py",
                     ".github/workflows/d68-d66-v041-1000-selfplay-train-audit.yml",
                     "research/d68/START.json"):
            if (out/name).exists():raise ValueError("D68 was already prepared; no replay")
    generated=[
        write(out,"scripts/d68_native_game.py",native_script(new,step)),
        write(out,"scripts/d68_games.py",games_script(new,step,weight,training,selfplay,audit)),
        write(out,"scripts/d68_train.py",train_script(step)),
        write(out,"scripts/d68_arena.py",arena_script(step)),
        write(out,".github/workflows/d68-d66-v041-1000-selfplay-train-audit.yml",
              workflow_script(new,step,weight,run,training,selfplay,audit))]
    request={
       "enabled":True,"request_id":"d68-20261108-d66-v041-selfplay1000-20runner",
       "baseline":new,"weights_sha256":weight,"optimizer_step":step,
       "search_variant":"d48_pvs_mate","search_sha256":CPP,
       "version":"FUKAJI v0.4.1","games":1000,"runners":20,
       "rounds":5,"compare_games":200,"formal_weights_adopt_after_verified_win":True}
    path=out/"research/d68/START.json";path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError("D68 request already exists")
    path.write_text(json.dumps(request,indent=2,ensure_ascii=False)+"\n")
    generated.append("research/d68/START.json")
    return generated

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--new",required=True)
    p.add_argument("--step",required=True,type=int)
    p.add_argument("--weights-sha",required=True)
    p.add_argument("--source-run",required=True,type=int)
    p.add_argument("--training-sha",required=True)
    p.add_argument("--selfplay-sha",required=True)
    p.add_argument("--audit-sha",required=True)
    p.add_argument("--out",type=pathlib.Path,default=ROOT)
    p.add_argument("--preview",action="store_true")
    a=p.parse_args()
    items=generate(a.out,new=a.new,step=a.step,weight=a.weights_sha,
                   run=a.source_run,training=a.training_sha,
                   selfplay=a.selfplay_sha,audit=a.audit_sha,
                   preview=a.preview)
    print(json.dumps({"schema":"d68-generated-once-v1",
                     "source_run":a.source_run,"files":items},ensure_ascii=False))
if __name__=="__main__":main()
