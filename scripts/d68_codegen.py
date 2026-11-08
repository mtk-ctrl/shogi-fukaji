"""Generate one bounded successor D68 run from verified D67 official D66 state.
All edits use frozen exact anchors and fail closed. Never launches itself, never
trains on audit games, and cannot create an unbounded recursive workflow.
"""
from __future__ import annotations
import argparse, json, re, pathlib, ast
from fukaji.online_learning import load_snapshot

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
