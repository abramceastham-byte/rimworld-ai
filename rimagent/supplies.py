"""What the colony has to build with, told apart the way a player sees it.

Models kept reading "ChunkSandstone 559" as stone to build with and planned
whole rooms in BlocksSandstone that nobody could ever build. Three things are
kept apart here:

  stock      allowed items in stockpiles and shelves: RimWorld's own resource
             readout (top right of the screen), from /resources/stored
  elsewhere  allowed items lying outside storage: still usable, colonists
             fetch them, but not in the readout
  forbidden  unusable until allow_items

and building materials ("stuff") are listed on their own, with stone chunks
explained as something to cut into blocks, not something to build with.
"""

from rimagent.construction import BUILDINGS, FLOORS, STUFF_BY_CATEGORY

STONES = ("Granite", "Sandstone", "Limestone", "Slate", "Marble")
BLOCKS_PER_CHUNK = 20  # one stonecutting job turns a chunk into 20 blocks
ALL_STUFF = tuple(m for names in STUFF_BY_CATEGORY.values() for m in names)


def stock_by_def(items, forbidden: bool) -> dict[str, int]:
    """How much of each item def is lying around, allowed or forbidden."""
    totals: dict[str, int] = {}
    for i in items:
        if i.is_forbidden == forbidden:
            totals[i.def_name] = totals.get(i.def_name, 0) + i.stack_count
    return totals


def usable_stuff(items) -> dict[str, int]:
    """Building materials the colonists can use now: allowed, anywhere on the map."""
    allowed = stock_by_def(items, False)
    return {m: allowed[m] for m in ALL_STUFF if allowed.get(m)}


def unlockable_stuff(items) -> dict[str, int]:
    """Building materials lying forbidden: one allow_items away from usable."""
    forbidden = stock_by_def(items, True)
    usable = usable_stuff(items)
    return {m: forbidden[m] for m in ALL_STUFF if forbidden.get(m) and not usable.get(m)}


def listed(totals: dict[str, int], limit: int = 24) -> str:
    """e.g. "Steel 480, WoodLog 220", biggest first."""
    top = sorted(totals.items(), key=lambda kv: -kv[1])[:limit]
    more = f", and {len(totals) - limit} more kinds" if len(totals) > limit else ""
    return ", ".join(f"{name} {n}" for name, n in top) + more if top else "nothing"


CHUNKS = tuple(f"Chunk{s}" for s in STONES)
CHUNK_NOTE = ("clutter, not building material: a stonecutter's table can cut them into stone blocks, "
              f"{BLOCKS_PER_CHUNK} per chunk")


class ItemLister:
    """listed(), with stone chunks set apart and explained beside them.

    The overview showed "ChunkSandstone 559" in the stock lines but not the
    line explaining chunks, and a model planned a bedroom "using ChunkSandstone,
    ChunkSlate, ChunkGranite". The full note goes with the first line listing
    chunks, a short one after: repeated stonecutter advice once made a model
    plan ten stonecutter's tables.
    """

    def __init__(self) -> None:
        self.explained = False

    def __call__(self, totals: dict[str, int]) -> str:
        chunks = {name: n for name, n in totals.items() if name in CHUNKS}
        if not chunks:
            return listed(totals)
        rest = {name: n for name, n in totals.items() if name not in CHUNKS}
        note = CHUNK_NOTE if not self.explained else "clutter, not building material"
        self.explained = True
        return (f"{listed(rest)}; " if rest else "") + f"stone chunks ({note}): {listed(chunks)}"


def how_to_get(stuff: str) -> str:
    """What to do when a material runs out, e.g. for a blueprint nobody can build."""
    if stuff.startswith("Blocks"):
        return (f"cut Chunk{stuff[len('Blocks'):]} into blocks at a stonecutter's table "
                f"(bill Make_Stone{stuff})")
    if stuff == "WoodLog":
        return "chop trees, or allow_items on forbidden wood"
    return f"allow_items on forbidden {stuff}, or mine it"


def describe_supplies(items, stored=None, tables=(), finished_research=frozenset(),
                      builds=()) -> list[str]:
    """The supplies section of the prompt.

    items: every item on the map (/map/things); stored: the items in storage
    (/resources/stored), or None when it couldn't be read; tables: built work
    tables, to say whether a stonecutter's table exists; builds: unfinished
    blueprints and frames (dicts with def_name, stuff, floor), whose material
    is already spoken for.
    """
    lines = []
    items_listed = ItemLister()
    if stored is not None:
        # Only stored things that are also loose items on the map: RIMAPI's
        # storage read also returns plants growing in stockpile cells, and
        # material already delivered into frames (at its old position).
        stored_ids = {i.thing_id for i in stored}
        in_storage = [i for i in items if i.thing_id in stored_ids]
        lines.append("Colony stock (allowed, in stockpiles and shelves, as the game's resource "
                     f"readout shows): {items_listed(stock_by_def(in_storage, False))}.")
        outside = [i for i in items if i.thing_id not in stored_ids]
        lines.append("Also allowed but outside storage (usable; colonists fetch it): "
                     f"{items_listed(stock_by_def(outside, False))}.")
    else:
        lines.append(f"Allowed items on the map (usable): {items_listed(stock_by_def(items, False))}.")
    lines.append(f"Forbidden, unusable until allow_items: {items_listed(stock_by_def(items, True))}.")

    stuff = usable_stuff(items)
    blocks = {m: n for m, n in stuff.items() if m.startswith("Blocks")}
    lines.append("Building materials (stuff) usable now: "
                 + (", ".join(f"{m} {n}" for m, n in stuff.items() if m not in blocks) or "no wood or steel")
                 + "; stone blocks: " + (", ".join(f"{m} {n}" for m, n in blocks.items()) or "none") + ".")
    locked = unlockable_stuff(items)
    if locked:
        # Offered for building all the same: a model with 300 forbidden wood
        # and no build_room on offer planned a room as 17 single walls.
        lines.append("Forbidden building materials, offered in the orders all the same: "
                     + ", ".join(f"{m} {n}" for m, n in locked.items())
                     + ". Include allow_items (what=all, or the material) in the same reply; blueprints only "
                       "wait for materials.")

    if builds:
        lines.append(describe_commitments(items, stored, builds))

    allowed = stock_by_def(items, False)
    chunks = {f"Chunk{s}": allowed[f"Chunk{s}"] for s in STONES if allowed.get(f"Chunk{s}")}
    if chunks:
        # One mention, only as strong as the situation needs: a model told in
        # four places to build a stonecutter's table planned ten of them.
        text = f"Stone chunks ({listed(chunks)}) are clutter, not building material; build in wood."
        if any(t.thing_def == "TableStonecutter" for t in tables):
            text += (f" Your stonecutter's table can cut them into blocks ({BLOCKS_PER_CHUNK} per chunk) "
                     f"with a bill such as Make_StoneBlocks{next(iter(chunks))[5:]}, if you want stone.")
        elif any(b.get("def_name") == "TableStonecutter" for b in builds):
            text += " Your stonecutter's table is planned; it can cut them into blocks once built."
        lines.append(text)
    return lines


def delivered_to_frames(items, stored) -> dict[str, int]:
    """Material already carried into frames under construction.

    RIMAPI's storage read walks every colony building that holds things, and a
    frame holds what was delivered to it: those entries are the stored things
    that aren't loose items on the map. Which frame holds what isn't reported.
    """
    if stored is None:
        return {}
    loose = {i.thing_id for i in items}
    return stock_by_def([i for i in stored if i.thing_id not in loose], False)


def commitments(items, stored, builds) -> tuple[dict, dict, dict]:
    """(material the builds need in all, already delivered to frames, still to come from stock)."""
    need = material_needs(builds, BUILDINGS, FLOORS)
    delivered = {m: n for m, n in delivered_to_frames(items, stored).items() if m in need}
    to_come = {m: n - delivered.get(m, 0) for m, n in need.items() if n > delivered.get(m, 0)}
    return need, delivered, to_come


def short_materials(items, stored, builds) -> set[str]:
    """Materials the unfinished builds need more of than the colony can use."""
    usable = stock_by_def(items, False)
    return {m for m, n in commitments(items, stored, builds)[2].items() if usable.get(m, 0) < n}


def describe_commitments(items, stored, builds) -> str:
    """What unfinished builds will still take from stock, and what that leaves.

    A model saw "Steel 380", planned a steel room and a steel table on top of
    it, and never counted the 100 Steel a half-built generator still needed.
    """
    need, delivered, to_come = commitments(items, stored, builds)
    usable = stock_by_def(items, False)
    text = f"Your unfinished builds need {listed(need)} in all"
    if delivered:
        text += f"; {listed(delivered)} of that is already delivered to their frames"
    if not to_come:
        return text + ". Nothing more is needed from stock; RIMAPI doesn't report how much work is left on a frame."
    left = [f"{m} {usable.get(m, 0) - n}" if usable.get(m, 0) >= n
            else f"{m} short by {n - usable.get(m, 0)} ({how_to_get(m)})" for m, n in to_come.items()]
    return (text + f"; still to come from stock: {listed(to_come)}. Left for new orders after that: "
            + ", ".join(left) + ". RIMAPI doesn't report how much work is left on a frame.")


def material_needs(blueprints: list[dict], buildings: dict, floors: dict) -> dict[str, int]:
    """Total material a list of submitted blueprints needs, e.g. {"WoodLog": 110}."""
    need: dict[str, int] = {}
    for bp in blueprints:
        if bp.get("floor"):
            spec = floors.get(bp["def_name"])
            costs = [spec.cost] if spec else []
        else:
            spec = buildings.get(bp["def_name"])
            costs = [(bp["stuff"], spec.stuff_count)] if spec and bp.get("stuff") and spec.stuff_count else []
            costs += list(spec.cost) if spec else []
        for name, count in costs:
            need[name] = need.get(name, 0) + count
    return need


def shortfall(need: dict[str, int], items) -> str:
    """"" when the colony has enough, else what's missing and how to get it."""
    allowed = stock_by_def(items, False)
    short = [(m, n, allowed.get(m, 0)) for m, n in need.items() if allowed.get(m, 0) < n]
    return "; ".join(f"needs {n} {m} but only {have} usable now: {how_to_get(m)}"
                     for m, n, have in short)
