#!/usr/bin/env bash
# Start sampler + server detached, then the replay waits and runs the pipeline.
# Kept separate from the calling shell session so nothing dies when it exits.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
setsid ./sampler.sh "$1" 0 > sampler.out 2>&1 < /dev/null &
echo "sampler: $1 (pid $!)"
setsid bash -c './server.sh > server.stdout 2>&1' < /dev/null &
echo "server: pid $! (log: server.log)"
