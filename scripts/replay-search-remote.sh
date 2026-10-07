#!/usr/bin/env bash
# Replays a two-line exchange that should trigger the search step, once per
# writer model, and shows the search call and the reference it produced.
# Usage: replay-search-remote.sh <writer model>...
set -euo pipefail
tree="$HOME/.cache/lilia-ci/worktrees/normal"
binary=$(ls -t "$tree"/build/linux/x86_64/*/lilia | head -1)
work=$(mktemp -d /tmp/lilia-search.XXXXXX)
trap 'rm -rf "$work"' EXIT
set -a
. "$HOME/.config/lilia/secrets.env"
set +a
cat >"$work/chat.jsonl" <<'EOF'
{"t": 0, "uid": 2003, "name": "Locietta", "text": "@莉莉娅 听说openai刚发了好几百个新的数学证明"}
{"t": 30, "uid": 2003, "name": "Locietta", "text": "真的假的 能看懂吗"}
EOF
for model in "$@"; do
    case $model in
        deepseek*) endpoint='' ;;
        *) endpoint='"writer_base_url": "https://api.minimaxi.com/anthropic", "writer_api_key": "$MINIMAX_API_KEY",' ;;
    esac
    cat >"$work/config.json" <<EOF
{"onebot": {"access_token": "\$LILIA_ONEBOT_TOKEN"},
 "model": {"api_key": "\$DEEPSEEK_API_KEY", "model": "$model", $endpoint "max_tokens": 1500},
 "chat": {"groups": [1]}, "data_dir": "$work", "log_level": "debug"}
EOF
    echo "#### writer: $model"
    "$binary" replay "$work/chat.jsonl" --config "$work/config.json" 2>"$work/log" || true
    { grep -E "search (for|reference)|search call|failed" "$work/log" || true; } | sed -E 's/^<[0-9]>//' | cut -c1-400
done
