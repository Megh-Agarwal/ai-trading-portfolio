#!/usr/bin/env bash
# Weekly rebalance cron wrapper — Ticket 6.3
#
# Cron entry (set after confirming server timezone is America/New_York):
#   30 6 * * 0 /home/ubuntu/ai-trading-portfolio/scripts/cron_weekly.sh
#
# On success: run_weekly.py uploads state.db + summary to S3 internally.
# On failure: this wrapper uploads the captured log to S3 for post-mortem.
# Lock file prevents duplicate runs within the same ISO week (defense in depth).

set -uo pipefail

# cron runs with a minimal PATH that excludes ~/.local/bin where uv is installed
export PATH="/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

REPO=/home/ubuntu/ai-trading-portfolio
LOG_DIR=$REPO/data/logs
DATE=$(date +%Y-%m-%d)
WEEK=$(date +%G-W%V)   # ISO week: %G = ISO year, %V = ISO week number
LOCK_FILE=/tmp/ai_trading_weekly_${WEEK}.lock
LOG_FILE=$LOG_DIR/weekly_run_${DATE}_$(date +%H%M%S).log

mkdir -p "$LOG_DIR"

# ── cron-level idempotency guard ──────────────────────────────────────────────
# Defense in depth against clock skew or duplicate cron entries.
# run_weekly.py has its own DB-backed guard; this is the outer shell layer.
if [[ -f "$LOCK_FILE" ]]; then
    echo "$(date -Iseconds) [SKIP] Lock file present for $WEEK — already ran this week" \
        >> "$LOG_FILE"
    exit 0
fi

cd "$REPO"

# Load .env so S3_BACKUP_BUCKET and API keys are available to child processes
set -a
# shellcheck source=/dev/null
source .env
set +a

echo "$(date -Iseconds) [START] Weekly rebalance date=$DATE week=$WEEK" | tee -a "$LOG_FILE"

# ── run the rebalance ─────────────────────────────────────────────────────────
EXIT_CODE=0
uv run python scripts/run_weekly.py --mode live 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo "$(date -Iseconds) [END] exit_code=$EXIT_CODE" | tee -a "$LOG_FILE"

# ── on failure: upload log to S3 for post-mortem ─────────────────────────────
# On success, run_weekly.py already uploaded state.db + summary internally.
# On failure it exits before reaching that code, so we upload the raw log here.
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "$(date -Iseconds) [BACKUP] Uploading failure log to S3" | tee -a "$LOG_FILE"
    DATE=$DATE EXIT_CODE=$EXIT_CODE LOG_FILE=$LOG_FILE \
    uv run python - <<'PYEOF' 2>&1 | tee -a "$LOG_FILE" || true
import os, sys
sys.path.insert(0, 'src')
from infra.s3_backup import upload_weekly_snapshot
date = os.environ['DATE']
log_file = os.environ['LOG_FILE']
exit_code = int(os.environ['EXIT_CODE'])
with open(log_file) as f:
    log_text = f.read()
upload_weekly_snapshot(
    date=date,
    db_path='data/state.db',
    log_text=log_text,
    summary_dict={'date': date, 'exit_code': exit_code, 'status': 'FAILED'},
)
print('Failure log uploaded to S3')
PYEOF
fi

# ── lock on success only ──────────────────────────────────────────────────────
# Failed runs are NOT locked — allows manual retry with --force flag.
# The lock is only against unintended duplicate triggers (cron, clock skew).
if [[ $EXIT_CODE -eq 0 ]]; then
    touch "$LOCK_FILE"
    echo "$(date -Iseconds) [LOCK] Created $LOCK_FILE" | tee -a "$LOG_FILE"
fi

exit $EXIT_CODE
