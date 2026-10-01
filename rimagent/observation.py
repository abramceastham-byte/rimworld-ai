"""Read-only operational observations. Optional endpoint failures stay explicit."""

from dataclasses import dataclass, field

import httpx

from rimagent.rimapi import RimApiError


@dataclass
class Operations:
    lines: list[str] = field(default_factory=list)
    targets: dict[str, list[dict]] = field(default_factory=dict)
    storage: dict | None = None  # /resources/storages/summary, for hints.storage_hint
    tagged: list[tuple[str, str]] = field(default_factory=list)  # (toolkit, line), for rimagent.modes

    def add(self, tag: str, line: str) -> None:
        """One observation line, tagged with the toolkit it belongs to."""
        self.lines.append(line)
        self.tagged.append((tag, line))


TARGET_TOOLKIT = {"mine": "wood_mining", "hunt": "hunting", "harvest": "farming"}


def read_operations(client, colonists, defs, map_id=0, fields=None):
    """fields: one description per growing zone ("Plant_Potato field (1,2)-(6,7),
    30 cells"), which /map/zones can't give since it has no locations."""
    result = Operations()

    def get(path, **params):
        try:
            return client._request("GET", "/api/v1/" + path, params={"map_id": map_id, **params})
        except (RimApiError, httpx.HTTPError, ValueError) as e:
            result.add("general", f"Unavailable observation {path}: {str(e)[:120]}")
            return None

    weather = get("map/weather")
    temperature = weather.get("temperature") if weather else None
    if weather:
        shown = f"{temperature:.0f}" if isinstance(temperature, (int, float)) else "?"
        result.add("general", f"Outdoors: {shown} C, {weather.get('weather', '?')}.")
    date = get("datetime")
    if date:
        result.add("general", "Date: " + date.get("datetime", "unknown"))
    farm = get("map/farm/summary")
    zones = (get("map/zones") or {}).get("zones") or []
    if isinstance(temperature, (int, float)):
        result.add("farming", growing_season(temperature))
    growing = [z for z in zones if "Growing" in (z.get("type") or "")]
    if colonists:
        result.add("farming", field_advice(sum(z.get("cells_count", 0) for z in growing), len(colonists),
                                           fields if fields is not None else
                                           [f"{z.get('label')}, {z.get('cells_count', 0)} cells" for z in growing]))
    if farm:
        result.add("farming", f"Farms: {farm.get('total_growing_zones', 0)} growing zones, "
                            f"{sum(z.get('cells_count', 0) for z in growing)} cells in all, "
                            f"{farm.get('total_plants', 0)} plants, "
                            f"{farm.get('total_infected_plants', 0)} infected.")
        crops = [c["plant_def_name"] for c in farm.get("crop_types", [])]
        for crop in farm.get("crop_types", []):
            name = crop["plant_def_name"]
            result.add("farming", describe_crop(crop, defs.plants.get(name, {})))
        # RIMAPI can't give a field's cells or its own growth (its per-zone
        # endpoint reads map_id as the zone id), so fields are found from
        # where each crop's plants stand.
        plants = client.get_plants(map_id)
        for crop, area, count in fields_from_plants(plants, crops):
            result.add("farming", f"Field of {crop} around {area}: {count} plants.")
        for crop in farm.get("crop_types", []):
            if crop.get("harvestable_plants", 0):
                # Plant rows carry no growth, so these are the crop's cells;
                # the harvest designation itself only takes ripe plants.
                result.targets.setdefault("harvest", []).extend(
                    {"name": crop["plant_def_name"], "x": p["position"]["x"], "z": p["position"]["z"]}
                    for p in plants if p.get("def_name") == crop["plant_def_name"])
    storage = get("resources/storages/summary")
    result.storage = storage
    if storage:
        result.add("storage", describe_storage(storage, [z for z in zones if "Stockpile" in (z.get("type") or "")]))
    power = get("map/power/info")
    if power:
        makers = len(power.get("produce_power_buildings") or [])
        users = len(power.get("consume_power_buildings") or [])
        if makers or users:
            result.add("power",
                f"Power: {makers} generators, {users} consumers; producing "
                f"{power.get('current_power', 0):.0f} W for {power.get('total_consumption', 0):.0f} W "
                f"demand; stored {power.get('currently_stored_power', 0):.0f}/"
                f"{power.get('total_power_storage', 0):.0f} Wd (map totals, not proof a building is connected).")
        else:
            result.add("power", "Power: no generators or powered buildings; electrical devices won't run.")
    rooms = get("map/rooms")
    if rooms:
        for room in rooms.get("rooms", []):
            beds = room.get("contained_beds_ids")
            if not beds or room.get("is_doorway"):  # a door cell is its own "room"
                continue
            if room.get("touches_map_edge"):  # the "room" RimWorld calls outdoors
                result.add("shelter", f"Beds outdoors, in no room: {beds}.")
            else:
                roof = room.get("open_roof_count") or 0
                result.add("shelter",
                    f"Bedroom with beds {beds}: {room.get('cells_count')} cells, "
                    f"{room.get('temperature', 0):.0f} C"
                    + (f", {roof} cells unroofed" if roof else ", roofed") + ".")
    windows = get("ui/windows")
    if windows:
        forced = [w["window_type"] for w in windows if w.get("force_pause")]
        if forced:
            result.add("general", "Forced pause windows: " + ", ".join(forced))
    ore = get("map/ore")
    if ore:
        width = ore.get("map_width", 0)
        if width:
            result.targets["mine"] = [
                {"name": name, "x": cell % width, "z": cell // width}
                for name, group in ore.get("ores", {}).items() for cell in group.get("cells", [])]
    # /map/animals 1.10 incorrectly puts z into y. Join reliable pawn positions
    # by ID instead of version-dependent coordinate guesses.
    animals = get("map/animals")
    if animals:
        positions = {p["id"]: p["position"] for p in get("map/pawns") or []}
        result.targets["hunt"] = []
        for animal in animals:
            pos = positions.get(animal["id"])
            # Wild animals only. Any faction affiliation is excluded.
            if not pos or animal.get("faction"):
                continue
            spec = defs.animals.get(animal.get("def"), {})
            result.targets["hunt"].append({"id": animal["id"], "name": animal.get("def"),
                "x": pos["x"], "z": pos["z"], "predator": spec.get("predator", "unknown"),
                "body_size": spec.get("base_body_size", "unknown")})
        hunters = []
        for pawn in colonists:
            if any(w.work_type == "Hunting" and w.priority > 0 for w in pawn.work_priorities):
                inventory = get("pawns/inventory", id=pawn.id)
                if inventory is not None:
                    weapons = [t["def_name"] for t in inventory.get("equipment", [])]
                    hunters.append(f"{pawn.name} ({', '.join(weapons) or 'unarmed'})")
        if hunters:
            result.add("hunting", "Hunters and their weapons: " + ", ".join(hunters)
                                + ". Unarmed hunters won't hunt; hunted animals may fight back.")
    anchor = next((c.position for c in colonists if c.position), None)
    # Exact nearby candidates, not entire ore grids or all wild animals.
    for kind, targets in result.targets.items():
        targets.sort(key=lambda t: abs(t["x"] - anchor.x) + abs(t["z"] - anchor.z) if anchor else 0)
        shown = representative_targets(targets)
        if targets:
            result.add(TARGET_TOOLKIT.get(kind, "general"), f"Nearest {kind} targets: " + "; ".join(describe_target(kind, t)
                                                                      for t in shown) + ".")
    return result


# RimWorld plant growth by temperature: full speed from 10 to 42 C, slowing
# to nothing at 0 C and at 58 C (Plant.GrowthRateFactor_Temperature).
GROW_MIN, GROW_FULL_LOW, GROW_FULL_HIGH, GROW_MAX = 0, 10, 42, 58
PLENTY_OF_STORAGE = 0.5  # under this share used, another stockpile isn't needed
FIELD_CELLS_PER_COLONIST = 10  # a recommendation shown to the model, never enforced


def field_advice(field_cells: int, colonists: int, fields: list[str] = ()) -> str:
    """How much growing area there is against the recommended amount, naming
    each field so the model can see its own fields meet the need."""
    target = FIELD_CELLS_PER_COLONIST * colonists
    text = (f"Recommended growing area: about {FIELD_CELLS_PER_COLONIST} cells per colonist, {target} for "
            f"{colonists}; you have {field_cells}")
    if fields:
        text += f" in {len(fields)} field{'s' if len(fields) > 1 else ''} ({'; '.join(fields)})"
    if field_cells < target:
        need = target - field_cells
        return text + (f", so one field of about {need} more cells "
                       "would help. Bigger isn't better: every cell must be sown and harvested by your few "
                       f"colonists, which takes them from other work. Fields beyond {2 * target} cells in all "
                       "are refused.")
    return text + (", which is enough: a plan step to make a field is already done. Don't add fields until you "
                   "have more colonists or food runs short; every extra cell is more sowing and harvesting.")


def growing_season(temperature: float) -> str:
    """Whether outdoor crops grow right now, and how fast."""
    if temperature <= GROW_MIN or temperature >= GROW_MAX:
        return (f"Growing season: NO, at {temperature:.0f} C outdoor crops don't grow at all; don't sow "
                "outdoors until it is between 0 and 58 C.")
    if GROW_FULL_LOW <= temperature <= GROW_FULL_HIGH:
        return f"Growing season: yes, crops grow at full speed at {temperature:.0f} C (10 to 42 C)."
    if temperature < GROW_FULL_LOW:
        rate = (temperature - GROW_MIN) / (GROW_FULL_LOW - GROW_MIN)
    else:
        rate = (GROW_MAX - temperature) / (GROW_MAX - GROW_FULL_HIGH)
    return (f"Growing season: yes but slow, crops grow at about {rate:.0%} speed at {temperature:.0f} C "
            "(full speed from 10 to 42 C).")


def fields_from_plants(plants: list[dict], crops: list[str], gap: int = 2):
    """(crop, bounding Rect, plant count) for each group of a sown crop's plants.

    Plants of the same crop within gap cells of each other belong to one field.
    """
    from rimagent.construction import Rect
    found = []
    for crop in crops:
        cells = {(p["position"]["x"], p["position"]["z"]) for p in plants if p.get("def_name") == crop}
        while cells:
            group, todo = set(), [cells.pop()]
            while todo:
                x, z = todo.pop()
                group.add((x, z))
                near = {(x + dx, z + dz) for dx in range(-gap, gap + 1) for dz in range(-gap, gap + 1)} & cells
                cells -= near
                todo.extend(near)
            xs, zs = [x for x, _ in group], [z for _, z in group]
            found.append((crop, Rect(min(xs), min(zs), max(xs), max(zs)), len(group)))
    return sorted(found, key=lambda f: -f[2])


def describe_storage(storage: dict, stockpiles: list[dict]) -> str:
    """How much storage there is, per stockpile, and whether more is needed."""
    used, total = storage.get("used_cells", 0), storage.get("total_cells", 0)
    if not total:
        return ("Storage: no stockpiles yet, so items lie where they fall and outdoors they deteriorate: a "
                "storage room (build_room purpose storage) with a stockpile over its inside is best.")
    sizes = ", ".join(f"{z.get('label', '?')} {z.get('cells_count', 0)} cells" for z in stockpiles)
    text = f"Storage: {used}/{total} cells used" + (f" across {len(stockpiles)} stockpiles ({sizes})" if sizes else "")
    if total and used < PLENTY_OF_STORAGE * total:
        return text + (f". {total - used} cells free: plenty of room, so don't create another stockpile; "
                       "stop expanding storage and reassess the colony's needs.")
    return text + "."


def describe_crop(crop: dict, plant_def: dict) -> str:
    """One crop's growth and readiness, from /map/farm/summary.

    RIMAPI 1.10 fills growth_progress_average with the SUM of every plant's
    growth (0-1 each) and never divides, so it is divided here. Its
    days_until_harvest truncates to a whole number and is always 0, so it is
    not used; the def's grow_days gives the time to full growth instead.
    """
    name, total = crop["plant_def_name"], crop.get("total_plants", 0)
    ready = crop.get("harvestable_plants", 0)
    growth = min(1.0, crop.get("growth_progress_average", 0) / total) if total else 0.0
    text = f"Crop {name}: {total} plants, average growth {growth:.0%}"
    grow_days = plant_def.get("grow_days")
    if isinstance(grow_days, (int, float)) and grow_days > 0 and growth < 1:
        text += f" (full growth takes {grow_days:g} days in good conditions)"
    if ready:
        return text + f"; {ready} ripe, harvested by colonists with Growing work."
    return text + "; none ripe yet, so there is nothing to harvest: let time pass."


def describe_target(kind: str, target: dict) -> str:
    """e.g. "Megasloth at (86,116), body size 4" (", PREDATOR" when it is one)."""
    text = f"{target['name']} at ({target['x']},{target['z']})"
    if kind == "hunt":
        size, predator = target.get("body_size"), target.get("predator")
        text += (f", body size {size:g}" if isinstance(size, (int, float)) else "")
        text += {True: ", PREDATOR", False: ""}.get(predator, ", predator unknown")
    return text


def representative_targets(targets: list[dict], limit: int = 8) -> list[dict]:
    """Prefer one nearby example of each resource/species before more of the same.

    Keep at least one of every kind even if a mod adds more than limit kinds.
    The full observation remains available for validation and execution.
    """
    shown, seen = [], set()
    for target in targets:
        if target["name"] not in seen:
            shown.append(target)
            seen.add(target["name"])
    for target in targets:
        if len(shown) >= limit:
            break
        if target not in shown:
            shown.append(target)
    return shown
