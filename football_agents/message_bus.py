"""TeamMessageBus — per-team channel for the Call skill (Phase 5c).

When a player invokes Call(message, audience), the runtime posts a Message
to this bus on behalf of the player's team. Teammates' EgocentricFilter
reads from the bus on each filter() call and attaches relevant messages
to Observation.heard_calls.

Thread safety: multiple PlayerAgent worker threads may post; main thread
reads via filter(). Use threading.Lock for the per-team list.

Lifetime: messages older than MESSAGE_LIFETIME_TICKS (default 30 ticks
~= 0.6s game time at 50 ticks/sec) are filtered out at read time.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from .perception import Vec2

# A "team" channel key as used by this bus — distinct from the perception
# layer's "team_a" / "team_b" naming. The bus uses the gfootball-side
# convention ("left" / "right") so the runtime can pass the same string
# straight from god_view without a translation layer.
TeamChannel = Literal["left", "right"]


@dataclass(frozen=True)
class Message:
    """A single Call posted to the team channel."""
    sender_player_id: int      # gfootball player_id (e.g., 8)
    sender_jersey: int         # display jersey number (could differ in future)
    sender_position: Vec2      # for distance-based audience filtering
    message: str               # Chinese text, e.g. "传给我"
    audience: str              # "team" | "nearby" (10m radius)
    tick_posted: int           # env tick when posted


@dataclass(frozen=True)
class HeardCall:
    """A Message annotated with how old it is from the listener's POV.

    Returned by read_for() so the prompt-renderer can show '刚刚' vs '0.5秒前'.
    """
    sender_player_id: int
    sender_jersey: int
    sender_position: Vec2
    message: str
    audience: str
    age_ticks: int             # current_tick - tick_posted


class TeamMessageBus:
    """Thread-safe per-team broadcast channel for the Call skill.

    Two channels: "left" and "right" (gfootball coordinate convention).
    Multiple PlayerAgent worker threads may post() concurrently; the
    main runner thread reads via read_for() once per env tick per agent.
    A single Lock guards both channels — Call traffic is sparse (handful
    per second across the whole match), so contention is negligible.
    """

    # ~0.6s game time at 50 env ticks/sec — long enough for a teammate's
    # next decision tick to pick it up, short enough that stale calls
    # don't pollute later observations.
    MESSAGE_LIFETIME_TICKS: int = 30

    # ~10m on the normalized field for 'nearby' audience.
    # gfootball x in [-1, 1] maps to ~105m, so 0.20 ~= 10.5m.
    NEARBY_RADIUS: float = 0.20

    def __init__(self) -> None:
        # Per-team storage. Keys: "left" | "right". Values: list of Message.
        self._channels: dict[str, list[Message]] = {"left": [], "right": []}
        self._lock = threading.Lock()

    def post(self, team: str, message: Message) -> None:
        """Post a message to the given team's channel. Thread-safe."""
        if team not in self._channels:
            raise ValueError(
                f"unknown team channel {team!r}; expected 'left' or 'right'"
            )
        with self._lock:
            self._channels[team].append(message)

    def read_for(
        self,
        team: str,
        listener_id: int,
        listener_position: Vec2,
        current_tick: int,
    ) -> list[HeardCall]:
        """Return messages on `team` channel that this listener can hear:
          - Skip own messages (sender_player_id == listener_id)
          - Skip stale (age >= MESSAGE_LIFETIME_TICKS)
          - For audience='team': always heard
          - For audience='nearby': only if distance < NEARBY_RADIUS

        Returns HeardCall objects (Messages annotated with age_ticks).
        """
        if team not in self._channels:
            raise ValueError(
                f"unknown team channel {team!r}; expected 'left' or 'right'"
            )
        # Snapshot under lock so we don't iterate a list being mutated by
        # a concurrent post(). Cheap (sparse traffic).
        with self._lock:
            snapshot = list(self._channels[team])

        heard: list[HeardCall] = []
        for msg in snapshot:
            # Skip own messages
            if msg.sender_player_id == listener_id:
                continue
            # Skip stale
            age = current_tick - msg.tick_posted
            if age >= self.MESSAGE_LIFETIME_TICKS:
                continue
            # Audience rules
            if msg.audience == "nearby":
                if listener_position.distance_to(msg.sender_position) >= self.NEARBY_RADIUS:
                    continue
            elif msg.audience != "team":
                # Unknown audience — be conservative and drop. (Sender-side
                # validation should have caught this; this is just defense.)
                continue
            heard.append(HeardCall(
                sender_player_id=msg.sender_player_id,
                sender_jersey=msg.sender_jersey,
                sender_position=msg.sender_position,
                message=msg.message,
                audience=msg.audience,
                age_ticks=age,
            ))
        return heard

    def prune_stale(self, current_tick: int) -> None:
        """Optional: drop messages older than MESSAGE_LIFETIME_TICKS to keep
        channel lists bounded across long matches. Safe to call any time.

        Not strictly required — read_for() filters stale at read time —
        but prevents unbounded list growth over a 90-min match.
        """
        with self._lock:
            for team in self._channels:
                self._channels[team] = [
                    m for m in self._channels[team]
                    if current_tick - m.tick_posted < self.MESSAGE_LIFETIME_TICKS
                ]
