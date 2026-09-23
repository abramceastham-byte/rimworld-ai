"""Read-only operational observations. Optional endpoint failures stay explicit."""

from dataclasses import dataclass, field

import httpx

from rimagent.rimapi import RimApiError


@dataclass
class Operations:
    lines: list[str] = field(default_factory=list)
    targets: dict[str, list[dict]] = field(default_factory=dict)


def read_operations(client, colonists, defs, map_id=0):
    result = Operations()

    def get(path, **params):
        try:
            return client._request("GET", "/api/v1/" + path, params={"map_id": map_id, **params})
        except (RimApiError, httpx.HTTPError, ValueError) as e:
            result.lines.append(f"Unavailable observation {path}: {str(e)[:120]}")
            return None

    weather = get("map/weather")
    if weather:
        temperature = weather.get("temperature")
        shown = f"{temperature:.0f}" if isinstance(temperature, (int, float)) else "?"
        result.lines.append(f"Outdoors: {shown} C, {weather.get('weather', '?')}. "
                            "Crops grow only in suitable temperatures, not just on fertile soil.")
    date = get("datetime")
    if date:
        result.lines.append("Date: " + date.get("datetime", "unknown"))
    farm = get("map/farm/summary")
    if farm:
        result.lines.append(f"Farms: {farm.get('total_growing_zones', 0)} zones, "
                            f"{farm.get('total_plants', 0)} plants, "
                            f"{farm.get('total_infected_plants', 0)} infected.")
        for crop in farm.get("crop_types", []):
            name = crop["plant_def_name"]
            result.lines.append(f"Crop {name}: {crop.get('total_plants', 0)} plants, "
                                f"{crop.get('harvestable_plants', 0)} harvestable; "
                                f"growth {crop.get('growth_progress_average', 0):.0%}.")
            if crop.get("harvestable_plants", 0):
                # Plant rows carry no growth, so these are the crop's cells;
                # the harvest designation itself only takes ripe plants.
                result.targets.setdefault("harvest", []).extend(
                    {"name": name, "x": p["position"]["x"], "z": p["position"]["z"]}
                    for p in client.get_plants(map_id) if p.get("def_name") == name)
    storage = get("resources/storages/summary")
    if storage:
        result.lines.append(f"Storage: {storage.get('used_cells', 0)}/{storage.get('total_cells', 0)} cells used.")
    power = get("map/power/info")
    if power:
        makers = len(power.get("produce_power_buildings") or [])
        users = len(power.get("consume_power_buildings") or [])
        if makers or users:
            result.lines.append(
                f"Power: {makers} generators, {users} consumers; producing "
                f"{power.get('current_power', 0):.0f} W for {power.get('total_consumption', 0):.0f} W "
                f"demand; stored {power.get('currently_stored_power', 0):.0f}/"
                f"{power.get('total_power_storage', 0):.0f} Wd (map totals, not proof a building is connected).")
        else:
            result.lines.append("Power: no generators or powered buildings; electrical devices won't run.")
    rooms = get("map/rooms")
    if rooms:
        for room in rooms.get("rooms", []):
            beds = room.get("contained_beds_ids")
            if not beds or room.get("is_doorway"):  # a door cell is its own "room"
                continue
            if room.get("touches_map_edge"):  # the "room" RimWorld calls outdoors
                result.lines.append(f"Beds outdoors, in no room: {beds}.")
            else:
                roof = room.get("open_roof_count") or 0
                result.lines.append(
                    f"Bedroom with beds {beds}: {room.get('cells_count')} cells, "
                    f"{room.get('temperature', 0):.0f} C"
                    + (f", {roof} cells unroofed" if roof else ", roofed") + ".")
    windows = get("ui/windows")
    if windows:
        forced = [w["window_type"] for w in windows if w.get("force_pause")]
        if forced:
            result.lines.append("Forced pause windows: " + ", ".join(forced))
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
            result.lines.append("Hunters and their weapons: " + ", ".join(hunters)
                                + ". Unarmed hunters won't hunt; hunted animals may fight back.")
    anchor = next((c.position for c in colonists if c.position), None)
    # Exact nearby candidates, not entire ore grids or all wild animals.
    for kind, targets in result.targets.items():
        targets.sort(key=lambda t: abs(t["x"] - anchor.x) + abs(t["z"] - anchor.z) if anchor else 0)
        shown = representative_targets(targets)
        if targets:
            result.lines.append(f"Nearest {kind} targets: " + "; ".join(describe_target(kind, t)
                                                                      for t in shown) + ".")
    return result


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
