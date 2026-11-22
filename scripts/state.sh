#!/usr/bin/env bash
# Keep the pipeline's SQLite database between runs as an encrypted GitHub release asset.
#
# The free hosted Postgres ran out of monthly compute twice, so the database is now a plain SQLite
# file: each run downloads it, works on it locally and uploads it again. Release assets are public
# on a public repo and the file holds a few days of cached article text, so it is encrypted with
# STATE_KEY (a repository secret, also in the home PC's .env) and never published in the clear.
#
#   scripts/state.sh pull <file> <asset>    download + decrypt (no-op if the asset does not exist)
#   scripts/state.sh push <file> <asset>    encrypt + upload, replacing the asset
#
# Needs: gh (authenticated), openssl, gzip, STATE_KEY.
set -euo pipefail

TAG=state
action=$1 file=$2 asset=$3
: "${STATE_KEY:?STATE_KEY is not set}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

ensure_release() {
  gh release view "$TAG" >/dev/null 2>&1 || gh release create "$TAG" --prerelease --title "Pipeline state" \
    --notes "Encrypted working files of the ingest pipeline. Not a software release; nothing here is readable without the key."
}

case "$action" in
  pull)
    if ! gh release download "$TAG" -p "$asset" -D "$tmp" 2>/dev/null; then
      echo "state: no asset $asset yet; starting empty"
      exit 0
    fi
    mkdir -p "$(dirname "$file")"
    openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass env:STATE_KEY -in "$tmp/$asset" | gunzip > "$file"
    echo "state: pulled $asset -> $file ($(du -h "$file" | cut -f1))"
    ;;
  push)
    [ -f "$file" ] || { echo "state: $file missing; nothing to push"; exit 0; }
    ensure_release
    gzip -c "$file" | openssl enc -aes-256-cbc -salt -pbkdf2 -iter 200000 -pass env:STATE_KEY -out "$tmp/$asset"
    gh release upload "$TAG" "$tmp/$asset" --clobber
    echo "state: pushed $file -> $asset ($(du -h "$tmp/$asset" | cut -f1) encrypted)"
    ;;
  *) echo "usage: $0 pull|push <file> <asset>" >&2; exit 2 ;;
esac
