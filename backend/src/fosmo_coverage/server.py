"""Dependency-free local HTTP service for Fosmo coverage confirmation."""

from __future__ import annotations

import argparse
import json
import re
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .tracker import CoverageFrameError, CoverageTracker, FrameObservation


FRAME_PATH = re.compile(r"^/coverage/sessions/([0-9a-fA-F-]+)/frames$")
SESSION_PATH = re.compile(r"^/coverage/sessions/([0-9a-fA-F-]+)$")
MAX_REQUEST_BYTES = 2 * 1024 * 1024


class CoverageSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, CoverageTracker] = {}
        self._lock = threading.Lock()

    def create(self) -> tuple[str, CoverageTracker]:
        session_id = str(uuid.uuid4())
        tracker = CoverageTracker()
        with self._lock:
            self._sessions[session_id] = tracker
        return session_id, tracker

    def get(self, session_id: str) -> CoverageTracker | None:
        with self._lock:
            return self._sessions.get(session_id)


class CoverageRequestHandler(BaseHTTPRequestHandler):
    store = CoverageSessionStore()
    server_version = "FosmoCoverage/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok", "service": "fosmo-coverage", "version": "1.0"})
            return
        match = SESSION_PATH.fullmatch(self.path)
        if match:
            tracker = self.store.get(match.group(1))
            if tracker is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "coverage session not found"})
            else:
                self._json(HTTPStatus.OK, tracker.status().to_dict())
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "route not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/coverage/sessions":
            session_id, tracker = self.store.create()
            self._json(
                HTTPStatus.CREATED,
                {"session_id": session_id, **tracker.status().to_dict()},
            )
            return

        match = FRAME_PATH.fullmatch(self.path)
        if not match:
            self._json(HTTPStatus.NOT_FOUND, {"error": "route not found"})
            return
        tracker = self.store.get(match.group(1))
        if tracker is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "coverage session not found"})
            return
        try:
            payload = self._request_json()
            observation = FrameObservation.from_payload(payload)
            status = tracker.ingest(observation)
        except (CoverageFrameError, json.JSONDecodeError, UnicodeDecodeError) as error:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error), **tracker.status().to_dict()})
            return
        self._json(HTTPStatus.OK, status.to_dict())

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.client_address[0]} - {format % args}")

    def _request_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise CoverageFrameError("invalid Content-Length") from error
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise CoverageFrameError(f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
        parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(parsed, dict):
            raise CoverageFrameError("request JSON must be an object")
        return parsed

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Fosmo full-circle coverage service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), CoverageRequestHandler)
    print(f"Fosmo coverage backend listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
