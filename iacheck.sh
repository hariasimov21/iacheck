#!/usr/bin/env sh
# Lanzador para macOS / Linux: ./iacheck.sh [scan|search|check|specs] ...
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$DIR/iacheck.py" "$@"
