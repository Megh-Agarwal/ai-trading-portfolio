"""Tests for src/infra/s3_backup.py — Ticket 6.2."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from infra.s3_backup import upload_weekly_snapshot


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "state.db"
    db.write_bytes(b"SQLite content")
    return db


def test_skips_when_bucket_env_not_set(tmp_db: Path) -> None:
    with patch.dict("os.environ", {}, clear=True):
        with patch("infra.s3_backup.boto3") as mock_boto3:
            upload_weekly_snapshot("2026-06-27", tmp_db, "log", {"date": "2026-06-27"})
    mock_boto3.client.assert_not_called()


def test_uploads_all_three_objects(tmp_db: Path) -> None:
    mock_s3 = MagicMock()
    with patch.dict("os.environ", {"S3_BACKUP_BUCKET": "test-bucket"}):
        with patch("infra.s3_backup.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            upload_weekly_snapshot(
                "2026-06-27",
                tmp_db,
                "log line 1\nlog line 2",
                {"date": "2026-06-27", "skipped": False},
            )

    mock_boto3.client.assert_called_once_with("s3")
    mock_s3.upload_file.assert_called_once_with(
        str(tmp_db), "test-bucket", "snapshots/2026-06-27/state.db"
    )
    put_calls = mock_s3.put_object.call_args_list
    assert len(put_calls) == 2

    log_call = put_calls[0]
    assert log_call == call(
        Bucket="test-bucket",
        Key="snapshots/2026-06-27/run.log",
        Body=b"log line 1\nlog line 2",
        ContentType="text/plain",
    )

    summary_call = put_calls[1]
    assert summary_call.kwargs["Bucket"] == "test-bucket"
    assert summary_call.kwargs["Key"] == "snapshots/2026-06-27/summary.json"
    assert summary_call.kwargs["ContentType"] == "application/json"
    parsed = json.loads(summary_call.kwargs["Body"])
    assert parsed["date"] == "2026-06-27"
    assert parsed["skipped"] is False


def test_s3_key_prefix_uses_date(tmp_db: Path) -> None:
    mock_s3 = MagicMock()
    with patch.dict("os.environ", {"S3_BACKUP_BUCKET": "my-bucket"}):
        with patch("infra.s3_backup.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            upload_weekly_snapshot("2025-12-31", tmp_db, "", {})

    mock_s3.upload_file.assert_called_once_with(
        str(tmp_db), "my-bucket", "snapshots/2025-12-31/state.db"
    )


def test_summary_dict_serialized_as_json(tmp_db: Path) -> None:
    import datetime

    mock_s3 = MagicMock()
    summary = {"date": datetime.date(2026, 6, 27), "value": 1_000_000.0}
    with patch.dict("os.environ", {"S3_BACKUP_BUCKET": "b"}):
        with patch("infra.s3_backup.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            upload_weekly_snapshot("2026-06-27", tmp_db, "", summary)

    body = mock_s3.put_object.call_args_list[1].kwargs["Body"]
    parsed = json.loads(body)
    assert parsed["value"] == 1_000_000.0
    assert parsed["date"] == "2026-06-27"  # datetime.date serialized via default=str


def test_boto3_error_propagates(tmp_db: Path) -> None:
    mock_s3 = MagicMock()
    mock_s3.upload_file.side_effect = RuntimeError("S3 unavailable")
    with patch.dict("os.environ", {"S3_BACKUP_BUCKET": "b"}):
        with patch("infra.s3_backup.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            with pytest.raises(RuntimeError, match="S3 unavailable"):
                upload_weekly_snapshot("2026-06-27", tmp_db, "", {})
