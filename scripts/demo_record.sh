#!/usr/bin/env bash
# Driver for docs/assets/demo.gif — regenerate from the repo root with:
#   asciinema rec --overwrite -c "bash scripts/demo_record.sh" /tmp/demo.cast
#   agg --font-size 16 --line-height 1.3 /tmp/demo.cast docs/assets/demo.gif
# Every output shown is real CLI output against a temp store.

set -euo pipefail
AM="$PWD/.venv/bin/agent-memory --data-dir /tmp/agent-memory-demo"
rm -rf /tmp/agent-memory-demo

type_line() {  # simulate typing
  printf '\033[1;32m$\033[0m '
  local s="$1"
  for ((i = 0; i < ${#s}; i++)); do
    printf '%s' "${s:i:1}"
    sleep 0.018
  done
  printf '\n'
}

comment() {
  printf '\033[2m%s\033[0m\n' "$1"
  sleep 0.6
}

run() {
  type_line "$1"
  shift
  "$@"
  sleep 2.2
  printf '\n'
}

comment "# Store once after a good answer"
run 'agent-memory remember "How do I reset my password?" "Go to Settings → Security → Reset Password." --type conversation --tags auth' \
    $AM remember "How do I reset my password?" "Go to Settings → Security → Reset Password." --type conversation --tags auth

comment "# Exact question → REPLAY (skip the LLM call)"
run 'agent-memory resolve "How do I reset my password?"' \
    $AM resolve "How do I reset my password?"

comment "# Paraphrase → RESTORE (inject as context, don't answer verbatim)"
run 'agent-memory resolve "I forgot my password"' \
    $AM resolve "I forgot my password"

comment "# Shared-word trap → NONE (refuses to misuse memory)"
run 'agent-memory remember "What payment methods do you support?" "Visa, Mastercard, and PayPal." --type conversation --tags billing' \
    $AM remember "What payment methods do you support?" "Visa, Mastercard, and PayPal." --type conversation --tags billing

run 'agent-memory resolve "Does the platform support two-factor authentication?"' \
    $AM resolve "Does the platform support two-factor authentication?"

sleep 2
