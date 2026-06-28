# Deployment Runbook

Operational guide for the live EC2 deployment. Covers initial setup context, routine redeploys, log access, manual runs, and rollback.

---

## Instance overview

| Property | Value |
|---|---|
| Provider | AWS EC2 |
| Instance type | t4g.small (ARM Graviton) |
| OS | Ubuntu 24.04 LTS |
| Region | us-east-2 |
| Timezone | America/New_York |
| Repo path | `~/ai-trading-portfolio/` |
| DB path | `~/ai-trading-portfolio/data/state.db` |
| Python toolchain | `uv` at `~/.local/bin/uv` |

The EC2 IP is fixed via an Elastic IP — find it in the AWS console under **EC2 → Elastic IPs**.

---

## Services

Three things run permanently on the instance:

| Service | What it does | How it runs |
|---|---|---|
| `ai-trading-api` | FastAPI on `127.0.0.1:8000` | systemd (auto-restart) |
| `nginx` | Reverse proxy on port 80 → port 8000 | systemd (auto-restart) |
| `cron` | Weekly rebalance at 6:30 AM ET every Sunday | crontab entry for `ubuntu` |

Check all three are healthy:

```bash
sudo systemctl status ai-trading-api
sudo systemctl status nginx
crontab -l
```

---

## SSH access

```bash
ssh -i ~/.ssh/ai-trading-ec2 ubuntu@<elastic-ip>
```

The key is `~/.ssh/ai-trading-ec2` (not the `.pem` file). The Elastic IP is in the AWS console.

---

## Redeploy after a code change

This is the standard flow for any change merged to `main` on the private repo:

```bash
# 1. SSH into EC2
ssh -i ~/.ssh/ai-trading-ec2 ubuntu@<elastic-ip>

# 2. Pull latest code
cd ~/ai-trading-portfolio
git pull

# 3. Restart the API service (nginx does not need a restart for Python-only changes)
sudo systemctl restart ai-trading-api

# 4. Confirm the API is healthy
curl http://localhost:8000/health
```

If `pyproject.toml` changed (new dependency added):

```bash
uv sync
sudo systemctl restart ai-trading-api
```

If nginx config changed (`/etc/nginx/sites-available/portfolio-api`):

```bash
sudo nginx -t          # validate config before reloading
sudo systemctl reload nginx
```

---

## Checking logs

### API service logs (most useful for debugging)

```bash
# Last 50 lines
sudo journalctl -u ai-trading-api -n 50

# Follow live
sudo journalctl -u ai-trading-api -f

# Errors only
sudo journalctl -u ai-trading-api -p err
```

### Weekly run logs

Each Sunday run writes a timestamped log to `data/logs/`:

```bash
ls -lt ~/ai-trading-portfolio/data/logs/
tail -100 ~/ai-trading-portfolio/data/logs/weekly_run_<date>.log
```

### nginx logs

```bash
sudo tail -50 /var/log/nginx/error.log
sudo tail -50 /var/log/nginx/access.log
```

### CloudWatch

Logs are shipped to log group `ai-trading/weekly-run`, stream `prod`. Access via the AWS console under **CloudWatch → Log groups**. A CloudWatch alarm fires an email alert if any run exits with `exit_code=1`.

### S3 backups

After every successful run, three files are uploaded to S3:
- `snapshots/{date}/state.db` — full SQLite snapshot
- `snapshots/{date}/run.log` — full run log
- `snapshots/{date}/summary.json` — NAV, turnover, cost summary

Access via the AWS console or `aws s3 ls s3://<bucket>/snapshots/` from the instance (auth via IAM role — no credentials needed).

---

## Manually triggering a rebalance

Use this when you need to run outside the Sunday cron schedule (e.g. after a hotfix, or to backfill a missed week):

```bash
cd ~/ai-trading-portfolio
source .env   # load API keys into current shell
uv run python scripts/run_weekly.py --date 2026-07-04
```

Omit `--date` to use today's date. Add `--force` to re-run a date that already has data (idempotency guard normally skips it).

The weekly cron script (`scripts/cron_weekly.sh`) uses an ISO-week lock file — running `run_weekly.py` directly bypasses the lock, which is intentional for manual runs.

---

## Rollback procedure

If a bad deploy breaks the API or cron:

### Option A — revert the code (preferred)

```bash
cd ~/ai-trading-portfolio
git log --oneline -5          # find the last good commit hash
git checkout <good-commit>    # detach HEAD to that commit
sudo systemctl restart ai-trading-api
curl http://localhost:8000/health
```

To return to tracking main after the fix is merged:

```bash
git checkout main
git pull
sudo systemctl restart ai-trading-api
```

### Option B — restore DB from S3 backup

If `state.db` is corrupted:

```bash
# Stop the API first so SQLite isn't accessed during restore
sudo systemctl stop ai-trading-api

# Download last known good snapshot from S3
aws s3 cp s3://<bucket>/snapshots/<last-good-date>/state.db \
    ~/ai-trading-portfolio/data/state.db

sudo systemctl start ai-trading-api
curl http://localhost:8000/health
```

---

## Environment variables

All secrets live in `~/ai-trading-portfolio/.env` (not committed). The systemd service loads this file via `EnvironmentFile=`. Required variables:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | LLM API calls |
| `FRED_API_KEY` | Macro data ingestion |
| `FINNHUB_API_KEY` | News ingestion (weekly refresh) |
| `GCP_PROJECT_ID` | GDELT BigQuery queries |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON |
| `S3_BACKUP_BUCKET` | S3 bucket name for weekly snapshots |
| `DASHBOARD_ORIGIN` | CORS allowed origin for the public dashboard |

If a variable is missing, the relevant pipeline step logs an error and continues (except `ANTHROPIC_API_KEY` — that halts the agent pipeline).

---

## Dashboard

The public dashboard is hosted on Hugging Face Spaces (separate repo, zero proprietary code). It calls this API over HTTP. No deploy step is needed on EC2 when the dashboard changes — HF Spaces rebuilds automatically when its own repo is updated.

If the dashboard shows stale data, the first thing to check is the API health endpoint:

```bash
curl http://localhost:8000/health
```

If `status` is `"stale"`, the cron missed a run — check the weekly run logs and CloudWatch.
