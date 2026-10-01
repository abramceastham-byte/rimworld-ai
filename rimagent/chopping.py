"""Trees the agent asked to be cut, and what became of them.

RIMAPI can't list designations, and its harvest order silently skips trees
that are too young (not HarvestableNow) or already marked, then reports
success without a count. So a model that orders chopping sees nothing come of
it and orders the same trees again: one run did the same rectangle three turns
running while the colony's wood sat at 6 for twenty turns.

This keeps the trees it asked for and checks them each turn: a tree gone from
the map was cut, and one still standing long after the order was most likely
too young (the game ignored the order) or nobody is free to cut it. RIMAPI's
plants carry no growth, so youth is only blamed once nothing at all has been
cut for a while: one run was told "most likely too young" ten turns running
while its trees fell one by one and its wood went from 0 to 518.
"""

import json
from pathlib import Path

from pydantic import BaseModel

STALE_TICKS = 5000  # ~2 in-game hours: a tree actually marked is usually cut by then


class ChopOrder(BaseModel):
    thing_id: int
    def_name: str
    x: int
    z: int
    tick: int  # game tick when the cut was ordered


class ChopTracker:
    def __init__(self, path: str | Path | None = None, read_only: bool = False):
        self.path = Path(path) if path else None
        self.read_only = read_only
        self.orders: list[ChopOrder] = []
        self.cut_since_last = 0
        self.last_cut_tick: int | None = None  # when a tree we ordered last fell (this session)
        self.wood: int | None = None           # allowed WoodLog on the map at the last reconcile
        self.wood_gained = 0
        if self.path and self.path.exists():
            self.orders = [ChopOrder.model_validate(o)
                           for o in json.loads(self.path.read_text(encoding="utf-8"))]

    def save(self) -> None:
        if self.path and not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([o.model_dump() for o in self.orders], indent=2),
                                 encoding="utf-8")

    def recent_ids(self, tick: int) -> set[int]:
        """Trees ordered recently enough that ordering them again would do nothing.
        Older ones may be ordered again: they may have grown old enough since."""
        return {o.thing_id for o in self.orders if tick - o.tick < STALE_TICKS}

    def add(self, trees, tick: int) -> None:
        ids = {t.thing_id for t in trees}
        self.orders = [o for o in self.orders if o.thing_id not in ids] + [
            ChopOrder(thing_id=t.thing_id, def_name=t.def_name, x=t.position.x, z=t.position.z, tick=tick)
            for t in trees]
        self.save()

    def reconcile(self, trees, tick: int | None = None, wood: int | None = None) -> None:
        """Forget trees no longer on the map: those were cut. wood is the allowed
        WoodLog on the map now, to report what the cutting brought in."""
        standing = {t.thing_id for t in trees}
        before = len(self.orders)
        self.orders = [o for o in self.orders if o.thing_id in standing]
        self.cut_since_last = before - len(self.orders)
        if self.cut_since_last:
            if tick is not None:
                self.last_cut_tick = tick
            self.save()
        if wood is not None:
            self.wood_gained = max(0, wood - self.wood) if self.wood is not None else 0
            self.wood = wood

    def describe(self, tick: int, cutters: list[str]) -> str | None:
        """One prompt line, or None when there is nothing to report."""
        if not self.orders and not self.cut_since_last:
            return None
        stale = [o for o in self.orders if tick - o.tick >= STALE_TICKS]
        text = (f"Tree cutting you ordered: {self.cut_since_last} cut since last turn (wood appears where "
                f"they fell); {len(self.orders)} still standing")
        cutting = self.last_cut_tick is not None and tick - self.last_cut_tick < STALE_TICKS
        if stale and cutting:
            text += (f", {len(stale)} of them ordered 2+ in-game hours ago; trees are still falling, so those "
                     "are waiting their turn: no new chop order is needed")
        elif stale:
            box = (min(o.x for o in stale), min(o.z for o in stale), max(o.x for o in stale), max(o.z for o in stale))
            text += (f", {len(stale)} of them ordered 2+ in-game hours ago, within ({box[0]},{box[1]})-"
                     f"({box[2]},{box[3]}), and none has fallen for 2+ hours: either they are too young (the game "
                     "silently ignores cut orders on young trees, so pick other trees) or your cutters are busy, "
                     "hungry or asleep (see what they are doing)")
        if self.wood is not None:
            text += (f". Allowed WoodLog on the map: {self.wood}"
                     + (f" (+{self.wood_gained} since last turn)" if self.wood_gained else ""))
        text += ". Ordering the same trees again changes nothing. "
        text += (f"PlantCutting is on for: {', '.join(cutters)}." if cutters else
                 "Nobody has PlantCutting on, so no tree will be cut: enable_work PlantCutting.")
        return text
