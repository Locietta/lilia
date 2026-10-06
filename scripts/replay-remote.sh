#!/usr/bin/env bash
# Runs `lilia replay` from the remote test tree (pixi run test-remote-linux
# builds it) with the live config and secrets, so API keys never leave the
# host. Usage: replay-remote.sh <log.jsonl relative to the repo> [persona.md]
set -euo pipefail
tree="$HOME/.cache/lilia-ci/worktrees/normal"
binary=$(ls -t "$tree"/build/linux/x86_64/*/lilia | head -1)
set -a
. "$HOME/.config/lilia/secrets.env"
set +a
args=(replay "$tree/$1" --config "$HOME/.config/lilia/config.json")
[ -n "${2:-}" ] && args+=(--persona "$tree/$2")
"$binary" "${args[@]}" 2>/tmp/lilia-replay.log
echo "--- model calls:"
grep -E "call for|failed|warn" /tmp/lilia-replay.log | sed -E 's/^<[0-9]>//' | tail -40
