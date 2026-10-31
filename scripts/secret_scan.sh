#!/usr/bin/env bash
# Scan tracked-to-be files for credentials and internal details. Prints file:line and TYPE only, never the match.
set -u
status=0
scan() {  # type regex
  local hits
  hits=$(grep -rInE --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=.pytest_cache --exclude-dir=.mypy_cache --exclude-dir=.ruff_cache --exclude-dir=*.egg-info --exclude='secret_scan.sh' --exclude=uv.lock -e "$2" . 2>/dev/null | grep -v 'a229c3948406bc2cf6eaf4873e662e70c6a04746' | cut -d: -f1,2 || true)   # public upstream commit pin
  if [ -n "$hits" ]; then echo "$hits" | sed "s/$/  [$1]/"; status=1; fi
}
scan openai-style-key 'sk-[A-Za-z0-9_-]{20,}'
scan google-api-key 'AIza[0-9A-Za-z_-]{30,}'
scan hf-token 'hf_[A-Za-z0-9]{20,}'
scan bearer-literal 'Bearer [A-Za-z0-9._-]{24,}'
scan long-hex '\b[0-9a-f]{40,}\b'
scan private-key '-----BEGIN [A-Z ]*PRIVATE KEY-----'
scan internal-path '/projects/|/home/[a-z]|speechteam|debonaire|iitm'
scan internal-host 'bodhanai-|node0[0-9]{2}\b|trycloudflare\.com/[a-z]'
scan ticket-noise '\bG-?[0-9]{3}[a-z]?\b'
[ $status -eq 0 ] && echo "secret scan: clean"
exit $status
