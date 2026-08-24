"""Dependency-free local HTTP service for Fosmo coverage confirmation."""

from __future__ import annotations

import argparse
import json
import re
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
    preview_directory: Path | None = None
    server_version = "FosmoCoverage/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok", "service": "fosmo-coverage", "version": "1.0"})
            return
        if self.path == "/preview/manifest":
            self._preview_manifest()
            return
        if self.path == "/preview/model.ply":
            self._preview_model()
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

    def _preview_manifest(self) -> None:
        try:
            payload, _ = self._preview_configuration()
        except FileNotFoundError as error:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
            return
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"invalid preview report: {error}"})
            return
        self._json(HTTPStatus.OK, payload)

    def _preview_model(self) -> None:
        try:
            _, model_path = self._preview_configuration()
        except FileNotFoundError as error:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
            return
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"invalid preview report: {error}"})
            return
        try:
            size = model_path.stat().st_size
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", 'attachment; filename="room_model.ply"')
            self.end_headers()
            with model_path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _preview_configuration(self) -> tuple[dict[str, Any], Path]:
        directory = self.preview_directory
        if directory is None:
            raise FileNotFoundError("no preview model is configured")
        explicit_manifest = directory / "preview_manifest.json"
        if explicit_manifest.is_file():
            payload = json.loads(explicit_manifest.read_text(encoding="utf-8"))
            model_file = payload["model_file"]
            if not isinstance(model_file, str) or Path(model_file).name != model_file:
                raise ValueError("model_file must be a filename within the preview directory")
            model_path = directory / model_file
            result = {
                "scan_id": payload["scan_id"],
                "frame_count": payload["frame_count"],
                "vertex_count": payload["vertex_count"],
                "triangle_count": payload["triangle_count"],
                "model_kind": payload.get("model_kind", "mesh"),
                "model_bytes": model_path.stat().st_size,
                "model_url": "/preview/model.ply",
            }
            if "dimensions_meters" in payload:
                result["dimensions_meters"] = payload["dimensions_meters"]
            if "height_meters" in payload:
                result["height_meters"] = payload["height_meters"]
        else:
            model_path = directory / "room_blender_mesh.ply"
            report = json.loads((directory / "fusion_report.json").read_text(encoding="utf-8"))
            fusion = report["fusion"]
            result = {
                "scan_id": report["scan_id"],
                "frame_count": report["frame_count"],
                "vertex_count": fusion["blender_mesh_vertices"],
                "triangle_count": fusion["blender_mesh_triangles"],
                "model_kind": "fused",
                "model_bytes": model_path.stat().st_size,
                "model_url": "/preview/model.ply",
            }
        if not model_path.is_file():
            raise FileNotFoundError("preview model is not ready")
        return result, model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Fosmo full-circle coverage service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--preview-directory",
        type=Path,
        help="directory containing room_blender_mesh.ply and fusion_report.json",
    )
    args = parser.parse_args()
    CoverageRequestHandler.preview_directory = (
        args.preview_directory.expanduser().resolve() if args.preview_directory else None
    )
    server = ThreadingHTTPServer((args.host, args.port), CoverageRequestHandler)
    print(f"Fosmo coverage backend listening on http://{args.host}:{args.port}")
    if CoverageRequestHandler.preview_directory:
        print(f"Serving iPhone preview from {CoverageRequestHandler.preview_directory}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
