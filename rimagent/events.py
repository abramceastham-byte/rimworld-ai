"""Collect letters and messages from RIMAPI's live event stream.

RIMAPI has no endpoint for reading letters (the envelopes on the right of the
screen) or messages (the short texts at the top left). It pushes them instead,
over a Server-Sent Events stream at /api/v1/events. EventListener keeps that
connection open on a background thread and stores what arrives; your loop
calls drain() each step to pick up everything new.

Only events that happen while the listener is connected are seen, so start it
before the loop.

    with EventListener() as events:
        ...
        for event in events.drain():
            print(event.kind, event.category, event.text)
"""

import json
import queue
import threading
from typing import Literal

import httpx
from pydantic import BaseModel

from rimagent.config import load_settings


class GameEvent(BaseModel):
    kind: Literal["letter", "message"]
    text: str      # a letter's title, or a message's full text
    category: str  # e.g. ThreatBig, ThreatSmall, NegativeEvent, PositiveEvent, NeutralEvent
    tick: int


class EventListener:
    def __init__(self, base_url: str | None = None):
        base_url = base_url or load_settings().rimapi_url
        self._url = f"{base_url}/api/v1/events"
        self._events: queue.Queue[GameEvent] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self.connected = threading.Event()

    def start(self, wait: float = 5.0) -> "EventListener":
        """Start listening, waiting up to `wait` seconds for the connection."""
        self._thread.start()
        self.connected.wait(wait)
        return self

    def stop(self) -> None:
        self._stop.set()

    def __enter__(self) -> "EventListener":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def drain(self) -> list[GameEvent]:
        """Everything received since the last drain(), oldest first."""
        drained = []
        while True:
            try:
                drained.append(self._events.get_nowait())
            except queue.Empty:
                return drained

    def _listen(self) -> None:
        # RIMAPI sends a heartbeat every few seconds, so a 30 s read timeout
        # only fires if the game has really stopped responding.
        timeout = httpx.Timeout(10.0, read=30.0)
        while not self._stop.is_set():
            try:
                with httpx.stream("GET", self._url, timeout=timeout) as response:
                    self.connected.set()
                    self._read_stream(response)
            except httpx.HTTPError:
                pass  # game closed or restarting: retry below
            self.connected.clear()
            self._stop.wait(2)

    def _read_stream(self, response: httpx.Response) -> None:
        # SSE format: "event: <type>" and "data: <json>" lines, then a blank line.
        event_type, data = None, []
        for line in response.iter_lines():
            if self._stop.is_set():
                return
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].strip())
            elif line == "":
                if event_type and data:
                    self._handle(event_type, "\n".join(data))
                event_type, data = None, []

    def _handle(self, event_type: str, data: str) -> None:
        if event_type not in ("letter_received", "message_received"):
            return  # heartbeats, game state snapshots, log lines
        try:
            payload = json.loads(data)
            if event_type == "letter_received":
                letter = payload["letter"]
                event = GameEvent(kind="letter", text=letter.get("label") or "",
                                  category=letter.get("def") or "", tick=payload.get("ticks", 0))
            else:
                message = payload["message"]
                event = GameEvent(kind="message", text=message.get("text") or "",
                                  category=message.get("def") or "", tick=payload.get("ticks", 0))
        except (ValueError, KeyError, TypeError):
            return  # malformed event: skip it rather than kill the listener
        self._events.put(event)
