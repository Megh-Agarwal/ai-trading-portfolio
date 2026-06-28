"""S3 backup for weekly rebalance snapshots — Ticket 6.2.

Uploads three objects per run under snapshots/{date}/:
  state.db     — full SQLite snapshot
  run.log      — captured log output from the run
  summary.json — structured dict returned by run_weekly

Auth: relies on the EC2 instance IAM role (no credentials in code).
Bucket name comes from S3_BACKUP_BUCKET env var.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)

_BUCKET_ENV_VAR = "S3_BACKUP_BUCKET"
_PREFIX = "snapshots"


def upload_weekly_snapshot(
    date: str,
    db_path: Path | str,
    log_text: str,
    summary_dict: dict,
) -> None:
    """Upload state.db, run log, and summary JSON to S3.

    Args:
        date: ISO date string (YYYY-MM-DD) — used as the S3 key prefix.
        db_path: Local path to data/state.db.
        log_text: Full captured log output from the run.
        summary_dict: Structured summary dict returned by run_weekly.

    Raises:
        botocore.exceptions.BotoCoreError: On any S3 upload failure.
    """
    bucket = os.environ.get(_BUCKET_ENV_VAR)
    if not bucket:
        logger.warning("S3_BACKUP_BUCKET not set — skipping backup")
        return

    s3 = boto3.client("s3")
    prefix = f"{_PREFIX}/{date}"

    s3.upload_file(str(db_path), bucket, f"{prefix}/state.db")
    logger.info("s3://%s/%s/state.db uploaded", bucket, prefix)

    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/run.log",
        Body=log_text.encode(),
        ContentType="text/plain",
    )
    logger.info("s3://%s/%s/run.log uploaded", bucket, prefix)

    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/summary.json",
        Body=json.dumps(summary_dict, indent=2, default=str).encode(),
        ContentType="application/json",
    )
    logger.info("s3://%s/%s/summary.json uploaded", bucket, prefix)
