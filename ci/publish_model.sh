#!/usr/bin/env bash
# Publish the trained SAC model to a Gitea release.
# Token resolution order:
#   1) $GITEA_TOKEN if set
#   2) the token embedded in the git remote URL (set at clone time)
# Optional env: GITEA_API, GITEA_REPO, MODEL_DIR, RELEASE_TAG
# Only publishes when training has finished (models/sac_smart_charger_final.zip exists).
set -euo pipefail

GITEA_API="${GITEA_API:-https://gitea.ai-charge.net/api/v1}"
GITEA_REPO="${GITEA_REPO:-AI-ChargeTechnologies/ML}"
MODEL_DIR="${MODEL_DIR:-models}"
RELEASE_TAG="${RELEASE_TAG:-sac-model-$(date +%Y%m%d-%H%M)}"

# Fall back to the token embedded in the git remote (no secret needs to be stored elsewhere).
if [[ -z "${GITEA_TOKEN:-}" ]]; then
  GITEA_TOKEN="$(git config --get remote.origin.url 2>/dev/null | sed -nE 's#https?://[^:]*:([^@]+)@.*#\1#p' || true)"
fi
: "${GITEA_TOKEN:?GITEA_TOKEN not set and not found in git remote URL}"

FINAL="$MODEL_DIR/sac_smart_charger_final.zip"
BEST="$MODEL_DIR/best_model.zip"

# Gate: only publish once training completed (final model written by train.py at the end).
if [[ ! -f "$FINAL" ]]; then
  echo "Training not finished: $FINAL not found — skipping publish."
  exit 0
fi

FILES=("$FINAL")
[[ -f "$BEST" ]] && FILES+=("$BEST")

echo "Creating release $RELEASE_TAG on $GITEA_REPO ..."
RID=$(curl -fsS -H "Authorization: token $GITEA_TOKEN" -H "Content-Type: application/json" \
  -X POST "$GITEA_API/repos/$GITEA_REPO/releases" \
  -d "$(python3 - "$RELEASE_TAG" <<'PY'
import json,sys
tag=sys.argv[1]
print(json.dumps({"tag_name":tag,"target_commitish":"main",
  "name":f"SAC model {tag}","body":"Automated final-model publish from Jenkins after 500k-step training.",
  "draft":False,"prerelease":False}))
PY
)" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")

for f in "${FILES[@]}"; do
  echo "Uploading $(basename "$f") ..."
  curl -fsS -H "Authorization: token $GITEA_TOKEN" -X POST \
    "$GITEA_API/repos/$GITEA_REPO/releases/$RID/assets?name=$(basename "$f")" \
    -F "attachment=@$f;type=application/zip" >/dev/null
done
echo "Published release $RELEASE_TAG with: ${FILES[*]##*/}"
