#!/bin/bash
# Record this user's scratch usage in BLOKUS_USAGE_FILE for train.py's quota guard.
# A scan takes about 30 s, so it is skipped if another job refreshed the file in
# the last 10 minutes.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
file="$BLOKUS_USAGE_FILE"
if [ -f "$file" ] && [ $(( $(date +%s) - $(stat -c %Y "$file") )) -lt 600 ]; then
    exit 0
fi
bytes=$(du -s --block-size=1 "$BLOKUS_USER_SCRATCH" 2>/dev/null | cut -f1)
[ -n "$bytes" ] || exit 1
printf '{"bytes": %s, "time": %s}\n' "$bytes" "$(date +%s)" > "$file.tmp" && mv "$file.tmp" "$file"
