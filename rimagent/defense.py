"""What the colony has for defense, and what actually counts as a defense.

RimWorld's "Need defenses" alert shows from about day 2 to day 5 while the
colony has no turret, trap, sandbags or barricade. With nothing saying so, a
model answered it with loose lines of wall that enclosed nothing, and with
barricades ordered on top of its own generator and bed. This block says what
clears the alert, what helps in a fight, and where attacks would come from.
"""

from rimagent.construction import Rect

# Security-tab buildings the agent can place, and what each is for.
DEFENSES = ("Barricade", "TrapSpike", "Turret_MiniTurret", "Sandbags")

ALERT_NOTES = {
    "Need defenses": ("clears once the colony has a Barricade, TrapSpike or turret (the defense toolkit); "
                      "walls don't count"),
}


def direction(dx: int, dz: int) -> str:
    """Compass direction of an offset, z north."""
    ns = "north" if dz > 0 else "south"
    ew = "east" if dx > 0 else "west"
    if abs(dz) > 2 * abs(dx):
        return ns
    if abs(dx) > 2 * abs(dz):
        return ew
    return f"{ns}-{ew}"


def defense_lines(contents: list[tuple[str, Rect, str]], doors: list[tuple[int, int]], enemies,
                  anchor: tuple[int, int] | None, armed: int, colonists: int, dormant: int = 0) -> list[str]:
    """The defense section for the prompt.

    contents: (def name, area, state) for buildings and blueprints; doors: door
    cells; enemies: rimapi.Enemy (asleep ones included); armed: colonists
    carrying a weapon; dormant: sleeping enemies the threats report, located
    or not (RIMAPI's pawn list leaves sleeping mechanoids out).
    """
    counts = {name: sum(1 for n, _, _ in contents if n == name) for name in DEFENSES}
    have = ", ".join(f"{n} {name}" for name, n in counts.items() if n)
    lines = ["Defense:",
             f"- Defenses built or planned: {have or 'none yet'}. Colonists armed: {armed} of {colonists}.",
             "- The Need defenses alert clears once the colony has a Barricade, a TrapSpike or a turret "
             "(sandbags too, but they take cloth). A line of wall doesn't count and protects nothing: raiders "
             "walk around it.",
             "- What helps early: 3 to 5 Barricades as cover 2 to 4 cells out from the door raiders would "
             "come to, facing the way they would come, so drafted colonists shoot from behind them; a "
             "TrapSpike or two on the path in (never next to another trap). A Turret_MiniTurret needs "
             "GunTurrets research, power and steel. Keep defenses out of rooms and off other buildings."]
    if doors:
        lines.append("- Doors raiders would come to: " + ", ".join(f"({x},{z})" for x, z in doors[:4]) + ".")
    asleep = [e for e in enemies if not e.active]
    if asleep and anchor:
        cx = round(sum(e.x for e in asleep) / len(asleep))
        cz = round(sum(e.z for e in asleep) / len(asleep))
        far = max(abs(cx - anchor[0]), abs(cz - anchor[1]))
        lines.append(f"- {len(asleep)} dormant enemies around ({cx},{cz}), {far} cells "
                     f"{direction(cx - anchor[0], cz - anchor[1])} of the base. They stay asleep unless "
                     "disturbed; if they wake they come from there, so that side is where defenses go.")
    elif dormant:
        lines.append(f"- {dormant} dormant enemies (see Threats) can't be located from here. They stay asleep "
                     "unless disturbed. Raids come in from the map edge, usually the nearest one.")
    elif anchor:
        lines.append("- No enemies on the map now. Raids come in from the map edge, usually the nearest one.")
    return lines
