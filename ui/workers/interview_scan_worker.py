"""
ui/workers/interview_scan_worker.py
===================================
Non-blocking QThread worker for downloading video links and scanning interview segments
using Whisper and Gemini AI in Template 3 (+ Interview).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from services.interview_scanner import InterviewScanner, InterviewSegment
from utils.logger import get_logger

logger = get_logger(__name__)


class InterviewScanWorker(QThread):
    """
    Background worker thread executing video downloading and interview timestamp scanning.
    Keeps the PySide6 UI fluid and responsive with real-time status and progress updates.
    """

    progress = Signal(int)                   # 0 - 100 percentage
    status_message = Signal(str)             # User-facing status text
    scan_finished = Signal(str, list)        # (video_path, list_of_interview_segments)
    scan_failed = Signal(str)                # Error description on failure

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        force_redownload: bool = False,
    ) -> None:
        super().__init__()
        self.url = url.strip()
        self.api_key = api_key
        self.force_redownload = force_redownload
        self.scanner = InterviewScanner(api_key=api_key)

    def run(self) -> None:
        """Run download and scan pipeline on worker thread."""
        logger.info("Starting InterviewScanWorker for URL: %s", self.url)
        self.status_message.emit("Connecting to video source...")
        self.progress.emit(5)

        try:
            def _progress_callback(pct: int, msg: str) -> None:
                self.progress.emit(pct)
                self.status_message.emit(msg)

            video_path, segments = self.scanner.scan_url(
                url=self.url,
                progress_cb=_progress_callback,
                force_redownload=self.force_redownload,
            )

            logger.info(
                "InterviewScanWorker succeeded: %s (%d segments detected)",
                video_path.name,
                len(segments),
            )
            self.progress.emit(100)
            self.status_message.emit(f"Scan complete: {len(segments)} interview segment(s) ready.")
            self.scan_finished.emit(str(video_path), segments)

        except Exception as exc:
            err_msg = str(exc)
            logger.error("InterviewScanWorker failed: %s", err_msg)
            self.status_message.emit(f"Scan failed: {err_msg[:60]}")
            self.scan_failed.emit(err_msg)
