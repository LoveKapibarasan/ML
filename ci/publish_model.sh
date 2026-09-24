#!/usr/bin/env bash
# Publish the trained SAC model to a Gitea release.
# Required env: GITEA_TOKEN
# Optional env: GITEA_API, GITEA_REPO, MODEL_DIR, RELEASE_TAG
# Only publishes when training has finished (models/sac_smart_charger_final.zip exists).
set -euo pipefail

GITEA_API="${GITEA_API:-https://gitea.ai-charge.net/api/v1}"
GITEA_REPO="${GITEA_REPO:-AI-ChargeTechnologies/ML}"
MODEL_DIR="${MODEL_DIR:-models}"
RELEASE_TAG="${RELEASE_TAG:-sac-model-$(date +%Y%m%d-%H%M)}"
: "${GITEA_TOKEN:?GITEA_TOKEN required}"

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
