"""
tests/test_token_telemetry.py
=============================
Unit & integration tests for Gemini API Token Usage & Cost Telemetry:
  - SmartSceneMatcher token counting & cost calculation
  - SQLite api_usage_logs persistence & lifetime aggregations
  - EpisodeCompiler & EpisodeCompilerWorker token stats propagation
  - StudioView UI telemetry badges & completion banners
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from PySide6.QtWidgets import QApplication, QMessageBox

from services.gemini_matcher import SmartSceneMatcher, NarrativeChunk
from services.scene_detector import Scene
from database.db import (
    init_db,
    record_api_usage,
    get_lifetime_token_stats,
    DatabaseManager,
    get_connection,
)
from services.episode_compiler import EpisodeCompileConfig, EpisodeCompiler, EpisodeCompileResult
from ui.workers.episode_worker import EpisodeCompilerWorker
from ui.views.studio_view import StudioView


@pytest.fixture(autouse=True)
def mock_qmessagebox():
    """Prevent blocking modal dialogs during automated tests."""
    with patch.object(QMessageBox, "exec", return_value=0), \
         patch.object(QMessageBox, "information", return_value=0), \
         patch.object(QMessageBox, "warning", return_value=0), \
         patch.object(QMessageBox, "critical", return_value=0):
        yield


@pytest.fixture
def clean_api_usage_db(tmp_path, monkeypatch):
    """Set up an isolated database for token telemetry tests."""
    test_db = tmp_path / "test_telemetry.db"
    monkeypatch.setattr("database.db._get_db_path", lambda: test_db)
    init_db()
    return test_db



def test_gemini_matcher_token_telemetry(monkeypatch):
    """Verify SmartSceneMatcher tracks usage_metadata and computes Flash USD cost."""
    matcher = SmartSceneMatcher(gemini_api_key="AIzaSyMockKeyForTestingPurposes12345")
    
    # Verify initial zero state
    initial_stats = matcher.get_usage_summary()
    assert initial_stats["total_tokens"] == 0
    assert initial_stats["api_calls_count"] == 0
    assert initial_stats["estimated_cost_usd"] == 0.0

    # Mock google.generativeai response
    fake_meta = MagicMock()
    fake_meta.prompt_token_count = 1200
    fake_meta.candidates_token_count = 300
    fake_meta.total_token_count = 1500

    fake_response = MagicMock()
    fake_response.usage_metadata = fake_meta
    fake_response.text = '[{"chunk_index": 0, "candidate_index": 0}]'

    chunks = [
        NarrativeChunk(
            chunk_index=0,
            start_sec=0.0,
            end_sec=5.0,
            duration_sec=5.0,
            text="The royal family gathered at Windsor castle.",
        )
    ]
    scenes = [
        Scene(
            scene_id=1,
            video_hash="abc",
            video_path=Path("dummy.mp4"),
            scene_index=1,
            start_sec=0.0,
            end_sec=10.0,
            duration_sec=10.0,
            description="Windsor castle exterior",
        )
    ]

    with patch("google.generativeai.configure"), \
         patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_model_inst = MagicMock()
        mock_model_inst.generate_content.return_value = fake_response
        mock_model_cls.return_value = mock_model_inst

        matched = matcher._match_with_gemini(chunks, scenes)
        assert len(matched) == 1
        assert matched[0].matched_scene == scenes[0]

    summary = matcher.get_usage_summary()
    assert summary["prompt_tokens"] == 1200
    assert summary["output_tokens"] == 300
    assert summary["total_tokens"] == 1500
    assert summary["api_calls_count"] == 1
    # Cost: 1200 * 0.075 / 1M + 300 * 0.30 / 1M = 0.00009 + 0.00009 = 0.00018
    assert abs(summary["estimated_cost_usd"] - 0.00018) < 1e-6



def test_sqlite_api_usage_persistence(clean_api_usage_db):
    """Test recording and aggregating lifetime API usage in SQLite."""
    stats_zero = get_lifetime_token_stats()
    assert stats_zero["total_calls"] == 0
    assert stats_zero["total_tokens"] == 0
    assert stats_zero["total_cost_usd"] == 0.0

    # Insert render 1
    row1 = record_api_usage(
        project_id=1,
        project_title="Episode 1",
        prompt_tokens=5000,
        output_tokens=1000,
        total_tokens=6000,
        estimated_cost_usd=0.000675,
    )
    assert row1 > 0

    # Insert render 2
    row2 = record_api_usage(
        project_id=2,
        project_title="Episode 2",
        prompt_tokens=7000,
        output_tokens=2000,
        total_tokens=9000,
        estimated_cost_usd=0.001125,
    )
    assert row2 > 0

    stats = get_lifetime_token_stats()
    assert stats["total_calls"] == 2
    assert stats["total_prompt_tokens"] == 12000
    assert stats["total_output_tokens"] == 3000
    assert stats["total_tokens"] == 15000
    assert abs(stats["total_cost_usd"] - 0.0018) < 1e-5

    # Check DatabaseManager static method alias
    db_stats = DatabaseManager.get_lifetime_token_stats()
    assert db_stats["total_tokens"] == 15000


def test_episode_worker_token_stats_signal(tmp_path):
    """Verify EpisodeCompilerWorker holds token_stats and emits token_usage_updated."""
    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"dummy")
    footage = tmp_path / "footage"
    footage.mkdir()
    out = tmp_path / "out.mp4"

    worker = EpisodeCompilerWorker(
        audio_path=audio,
        footage_path=footage,
        output_path=out,
    )

    received_stats = []
    worker.token_usage_updated.connect(lambda stats: received_stats.append(stats))

    mock_result = EpisodeCompileResult(
        output_path=out,
        project_id=1,
        total_clips=2,
        duration_sec=10.0,
        width=1920,
        height=1080,
        elapsed_sec=1.5,
        token_stats={
            "prompt_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "api_calls_count": 1,
            "estimated_cost_usd": 0.000135,
        },
    )

    with patch("ui.workers.episode_worker.EpisodeCompiler") as MockCompiler:
        mock_inst = MagicMock()
        mock_inst.compile.return_value = mock_result
        MockCompiler.return_value = mock_inst

        worker.run()

    assert worker.token_stats["total_tokens"] == 1200
    assert len(received_stats) == 1
    assert received_stats[0]["total_tokens"] == 1200


def test_studio_view_token_telemetry_ui(qtbot, clean_api_usage_db, monkeypatch):
    """Verify StudioView UI elements: status badge, lifetime stats, and token usage banner."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestValidKey1234567890")

    # Record some lifetime usage
    record_api_usage(
        project_id=1,
        project_title="Initial Test",
        prompt_tokens=4000,
        output_tokens=1000,
        total_tokens=5000,
        estimated_cost_usd=0.0006,
    )

    view = StudioView()
    qtbot.addWidget(view)

    # By default, combo_format is Template 1 (index 0)
    assert view.combo_format.currentIndex() == 0
    assert view.btn_api_dashboard.isHidden()

    # Switch to Template 2 (Episode Scene Matching)
    view.combo_format.setCurrentIndex(1)
    assert not view.btn_api_dashboard.isHidden()
    assert "Gemini AI" in view.lbl_gemini_status.text()
    assert "Matcher Active" in view.lbl_gemini_status.text()
    assert "5,000 tokens" in view.lbl_gemini_status.text()

    # Verify banner initial state
    assert view.lbl_token_usage_banner.isHidden()

    # Trigger token_usage_updated
    dummy_stats = {
        "prompt_tokens": 3000,
        "output_tokens": 500,
        "total_tokens": 3500,
        "api_calls_count": 1,
        "estimated_cost_usd": 0.000375,
    }
    view._on_token_usage_updated(dummy_stats)

    assert not view.lbl_token_usage_banner.isHidden()
    assert "3,500 tokens" in view.lbl_token_usage_banner.text()
    assert "In: 3,000" in view.lbl_token_usage_banner.text()
    assert "Out: 500" in view.lbl_token_usage_banner.text()

