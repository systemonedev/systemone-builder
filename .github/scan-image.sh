#!/usr/bin/env bash
# Fail if an image contains credential files, key/token patterns or secret-looking env vars.
# Copies application and home directories out of a stopped container and scans them with the
# host's tools, so it works the same for Debian, Alpine/BusyBox and non-root images.
set -euo pipefail
image="$1"
work=$(mktemp -d)
cid=$(docker create "$image")
trap 'docker rm -f "$cid" >/dev/null; rm -rf "$work"' EXIT
for d in /app /opt/kenning /workspace /root /home; do
  docker cp "$cid:$d" "$work/" 2>/dev/null || true
done
files=$(find "$work" \( -name node_modules -o -name site-packages \) -prune -o -type f \
  \( -name ".env" -o -name ".env.*" -o -name "*.pem" -o -name "*.key" -o -name "id_rsa*" \
     -o -name "*.p12" -o -name "system_specification_sheet.md" \) -not -name ".env.example" -print)
dirs=$(find "$work" -name .git -prune -print)
hits=$(grep -rIlE --exclude-dir=node_modules --exclude-dir=site-packages \
  'hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|apikey_[0-9a-f]{8,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY' \
  "$work" || true)
# GPG_KEY is the public fingerprint official Python images use to verify their download.
env_vars=$(docker image inspect "$image" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -iE '(key|token|secret|password)=' | grep -vE '^GPG_KEY=[0-9A-F]{40}$' || true)
scanned=$(find "$work" -type f | wc -l)
if [ -n "$files$dirs$hits$env_vars" ]; then
  echo "::error::credential material found in $image"
  printf 'files:\n%s\n.git:\n%s\ncontents:\n%s\nenv:\n%s\n' "$files" "$dirs" "$hits" "$env_vars"
  exit 1
fi
[ "$scanned" -gt 0 ] || { echo "::error::nothing was scanned in $image"; exit 1; }
echo "clean: $scanned application files scanned in $image; no credential files, key patterns or secret env vars"
