#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hook_path="$repo_root/.git/hooks/commit-msg"

cat > "$hook_path" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
exec python3 "$repo_root/tools/check_harness_commit.py" "$1"
EOF

chmod +x "$hook_path"
echo "Installed MiniSolver commit-msg hook at $hook_path"
