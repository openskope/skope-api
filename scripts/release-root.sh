#!/bin/sh
# Print the physical path of a pin file's release_root (PIN-001, TXN-011).
# Usage: scripts/release-root.sh deploy/releases/<environment>.yml
# A relative release_root is resolved against the repository root.
set -eu
pin="$1"
test -f "$pin" || { echo "pin file not found: $pin" 1>&2; exit 2; }
root=$(sed -n 's/^release_root:[[:space:]]*//p' "$pin" | sed 's/[[:space:]]*#.*$//; s/^["'\'']//; s/["'\'']$//')
test -n "$root" || { echo "$pin has no release_root" 1>&2; exit 2; }
case "$root" in
  *://*) echo "$pin: object-storage release roots are not supported yet (TXN-012): $root" 1>&2; exit 2;;
  /*) ;;
  *) root="$(cd "$(dirname "$0")/.." && pwd)/$root";;
esac
test -d "$root" || { echo "$pin: release_root is not a directory: $root" 1>&2; exit 2; }
# The resolved physical path, never a symlink, is the mount source (TXN-011).
cd "$root" && pwd -P
