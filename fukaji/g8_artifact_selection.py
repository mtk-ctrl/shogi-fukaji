"""Read-only, SHA-pinned G8 candidate selection from existing GitHub artifacts.

No games, training, dispatch, engine changes, or promotion. The bearer token is
used ONLY against api.github.com; never sent to the artifact redirect host.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from .generation8_selection import (
    OLD_G7_ZIP_SHA, SOURCE_AGGREGATE_SHA, SOURCE_RUN, SEED, execute,
)

REPO = "mtk-ctrl/shogi-fukaji"
AGGREGATE_ID = 12783855111
AGGREGATE_NAME = "fukaji-g7-400-aggregate-40610843978"
OLD_RUN = 40434466661
OLD_ID = 12702814791
OLD_NAME = "fukaji-generation7-research-40434466661"
MAX_ZIP_BYTES = 16 * 1024 * 1024


def expected_request(enabled=False):
    return {
        "enabled": enabled,
        "research_only": True,
        "training_allowed": False,
        "adoption_allowed": False,
        "source_run_id": SOURCE_RUN,
        "source_artifact_id": AGGREGATE_ID,
        "source_artifact_sha256": SOURCE_AGGREGATE_SHA,
        "old_run_id": OLD_RUN,
        "old_artifact_id": OLD_ID,
        "old_artifact_sha256": OLD_G7_ZIP_SHA,
        "selection_seed": SEED,
    }


def validate_request(cfg):
    if cfg == expected_request(False):
        return False
    if cfg == expected_request(True):
        return True
    raise ValueError("G8 request differs from frozen read-only protocol")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def allowed_artifact_redirect(url):
    u = urlsplit(url)
    host = (u.hostname or "").lower()
    if u.scheme != "https" or u.username or u.password or u.port not in (None, 443):
        return False
    return any(host.endswith(suffix) and host != suffix[1:] for suffix in (
        ".blob.core.windows.net", ".githubusercontent.com",
        ".actions.githubusercontent.com",
    ))


def api_json(path, token):
    req = Request("https://api.github.com/repos/" + REPO + path,
                  headers={"Authorization": "Bearer " + token,
                           "Accept": "application/vnd.github+json",
                           "X-GitHub-Api-Version": "2022-11-28",
                           "User-Agent": "fukaji-g8-readonly"})
    with urlopen(req, timeout=40) as response:
        return json.load(response)


def artifact_bytes(artifact_id, name, run_id, expected_sha, token):
    meta = api_json("/actions/artifacts/" + str(artifact_id), token)
    if (meta.get("id") != artifact_id or meta.get("name") != name or
            meta.get("expired") is not False or
            meta.get("digest") != "sha256:" + expected_sha or
            (meta.get("workflow_run") or {}).get("id") != run_id or
            not 0 < meta.get("size_in_bytes", 0) <= MAX_ZIP_BYTES):
        raise ValueError("artifact identity/digest/expiry/run mismatch: " + name)
    req = Request("https://api.github.com/repos/" + REPO +
                  "/actions/artifacts/" + str(artifact_id) + "/zip",
                  headers={"Authorization": "Bearer " + token,
                           "Accept": "application/vnd.github+json",
                           "User-Agent": "fukaji-g8-readonly"})
    try:
        with build_opener(NoRedirect()).open(req, timeout=40):
            raise ValueError("expected GitHub artifact redirect, got direct response")
    except HTTPError as error:
        if error.code not in (302, 303, 307, 308):
            raise
        location = error.headers.get("Location")
    if not location or not allowed_artifact_redirect(location):
        raise ValueError("untrusted artifact redirect host")
    # Deliberately create a NEW request with NO Authorization header.
    # Never print the signed URL (it can contain a short-lived secret).
    public_req = Request(location, headers={"User-Agent": "fukaji-g8-readonly"})
    with urlopen(public_req, timeout=90) as response:
        chunks, total = [], 0
        while True:
            block = response.read(65536)
            if not block:
                break
            total += len(block)
            if total > MAX_ZIP_BYTES:
                raise ValueError("artifact ZIP too large")
            chunks.append(block)
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("downloaded artifact ZIP SHA mismatch: " + name)
    return raw


def run(request, out):
    cfg = json.loads(Path(request).read_text(encoding="utf-8"))
    enabled = validate_request(cfg)
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("enabled=" + ("true" if enabled else "false") + "\n")
    if out is None:
        print("G8 selection request valid; enabled:", enabled)
        return None
    if not enabled:
        raise ValueError("disabled G8 selection request")
    if os.environ.get("GITHUB_REPOSITORY") != REPO:
        raise ValueError("wrong GitHub repository")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("missing GitHub Actions read token")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source-aggregate-original.zip"
    old = root / "old-g7-original.zip"
    source.write_bytes(artifact_bytes(AGGREGATE_ID, AGGREGATE_NAME,
                                      SOURCE_RUN, SOURCE_AGGREGATE_SHA, token))
    old.write_bytes(artifact_bytes(OLD_ID, OLD_NAME, OLD_RUN, OLD_G7_ZIP_SHA, token))
    result = execute(source, old, root / "selected")
    receipt = {
        "status": "read_only_actual_legal_replay_verified",
        "source_run_id": SOURCE_RUN,
        "source_artifact_id": AGGREGATE_ID,
        "source_artifact_zip_sha256": SOURCE_AGGREGATE_SHA,
        "old_run_id": OLD_RUN,
        "old_artifact_id": OLD_ID,
        "old_artifact_zip_sha256": OLD_G7_ZIP_SHA,
        "all_artifact_metadata_and_bytes_verified": True,
        "training_executed": False,
        "model_promoted": False,
        "selection": result,
    }
    (root / "verification_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--out")
    args = p.parse_args()
    run(args.request, args.out)


if __name__ == "__main__":
    main()
