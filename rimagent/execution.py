"""Execute one order, distinguish unchanged, verified and submitted outcomes."""

import json

from rimagent.choices import action_target, incapable_work
from rimagent.hints import ONE_AT_A_TIME
from rimagent.rimapi import RimApiError


def execute(client, action, blueprints, zones, operations, learned_incapable=None, chops=None, mines=None):
    """Carry out one order against freshly read state, and say what it did.

    Returns "unchanged: ..." when the game already matches, or what was
    verified or merely submitted. Raises ValueError/RimApiError on failure.
    learned_incapable collects work RIMAPI refused, for later turns' choices;
    chops (a ChopTracker) remembers which trees were ordered cut, mines (a
    MineTracker) which rock was marked for mining.
    Does not advance the game clock.
    """
    client.begin_snapshot()
    client.submitted_blueprints = []
    kind = action.action
    target = json.dumps(action_target(action), separators=(",", ":"))
    detail = ""
    if kind in ("enable_work", "disable_work"):
        pawn = next((p for p in client.get_colonists() if p.name == action.colonist), None)
        if pawn is None:
            raise ValueError(f"colonist {action.colonist} no longer exists")
        enabled = any(w.work_type == action.work_type and w.priority > 0 for w in pawn.work_priorities)
        wanted = kind == "enable_work"
        if enabled == wanted:
            return (f"unchanged: {target} already {'enabled' if wanted else 'disabled'}"
                    + ("; asking again does nothing: if that work isn't getting done, look for "
                       "missing materials, too little time advanced, or more urgent jobs" if wanted else ""))
        if wanted and action.work_type in incapable_work(pawn, learned_incapable):
            raise ValueError(f"{pawn.name} is incapable of {action.work_type}")
        try:
            (client.enable_work if wanted else client.disable_work)(pawn.id, action.work_type)
        except RimApiError as error:
            # RIMAPI lists only enabled work, so this refusal is the only sign.
            if "disabled work type" not in str(error) or learned_incapable is None:
                raise
            learned_incapable.setdefault(pawn.name, set()).add(action.work_type)
            raise ValueError(f"{pawn.name} is incapable of {action.work_type} "
                             "(RimWorld refuses it); removed from the choices") from error
        current = next(p for p in client.get_colonists() if p.id == pawn.id)
        actual = any(w.work_type == action.work_type and w.priority > 0 for w in current.work_priorities)
        if actual != wanted:
            raise ValueError(f"work setting did not change: {target}")
        detail = "verified work setting"
    elif kind == "set_research":
        detail = client.set_research(action.project)
        if detail.startswith("unchanged"):
            return (f"{target}: {detail}; it carries on by itself while time advances, "
                    "so asking again does nothing")
        current = client.get_research().current
        if not current or current.name != action.project:
            raise ValueError(f"research selection was not applied: {action.project}")
        detail = f"{detail}; verified selected project (not completed)"
    elif kind == "set_bill":
        tables = client.get_work_tables()
        table = next((t for t in tables if t.id == action.building_id), None)
        if table is None:
            raise ValueError(f"no built work table {action.building_id}")
        existing = next((b for b in table.bills if b.recipe_def_name == action.recipe_def_name), None)
        if existing and existing.repeat_mode == "TargetCount" and existing.target_count == action.target_count and not existing.suspended:
            return f"unchanged: {target} already configured"
        client.set_target_bill(action.building_id, action.recipe_def_name, action.target_count)
        table = next((t for t in client.get_work_tables() if t.id == action.building_id), None)
        if not table or not any(b.recipe_def_name == action.recipe_def_name and
                b.repeat_mode == "TargetCount" and b.target_count == action.target_count and not b.suspended for b in table.bills):
            raise ValueError(f"bill not verified after request: {target}")
        detail = "verified bill configuration; production requires time, ingredients and worker"
    elif kind in ("create_growing_zone", "create_stockpile"):
        rect = action.rect()
        # Reconcile externally removed zones before treating a repeat as unchanged.
        live = {z.id for z in client.get_zones()}
        for z in zones.items:
            if z.zone_id in live and z.area() == rect and ((kind == "create_stockpile" and z.kind == "stockpile") or
                    (kind == "create_growing_zone" and z.kind == "growing" and z.label == f"{action.plant} field")):
                return f"unchanged: {target}, zone {z.zone_id} already exists (crop edits are not observable reliably)"
        if kind == "create_growing_zone":
            zone_id = client.create_growing_zone(action.plant, rect)
            zones.add("growing", zone_id, f"{action.plant} field", rect)
        else:
            zone_id = client.create_stockpile(rect)
            zones.add("stockpile", zone_id, "stockpile", rect)
        actual = next((z for z in client.get_zones() if z.id == zone_id), None)
        if actual is None or actual.cells_count != rect.cells:
            raise ValueError(f"zone {zone_id} creation incomplete: expected {rect.cells} cells; "
                             f"observed {actual.cells_count if actual else 'missing'}; refresh before retry")
        detail = f"verified zone {zone_id}, {rect.cells} cells; crop/geometry recorded locally"
    elif kind == "allow_items":
        rect = action.rect()
        before = [t for t in client.get_items() if t.is_forbidden and rect.contains(t.position.x, t.position.z)]
        if not before:
            return f"unchanged: {target}, no forbidden items remain"
        count = client.allow_items(rect)
        remaining = [t for t in client.get_items() if t.is_forbidden and t.thing_id in {i.thing_id for i in before}]
        if remaining:
            raise ValueError(f"allow_items partially applied: {len(remaining)}/{len(before)} still forbidden")
        detail = f"verified {count} stacks allowed"
    elif kind == "chop_trees":
        rect = action.rect()
        tick = client.get_state().game_tick
        ordered = chops.recent_ids(tick) if chops else set()
        # Read before ordering: the first order clears the cached tree list.
        already = sum(1 for t in client.get_trees()
                      if t.thing_id in ordered and rect.contains(t.position.x, t.position.z))
        done, total = client.chop_trees(rect, skip=ordered)
        if not done:
            return (f"unchanged: {target}: all {total} trees there were already ordered cut; "
                    "see 'Tree cutting you ordered' for what became of them")
        if chops:
            chops.add(done, tick)
        notes = [f"{already} there were already ordered"] if already else []
        more = total - already - len(done)
        if more > 0:
            notes.append(f"{more} more there beyond the {len(done)} per order: order again for those")
        detail = (f"ordered {len(done)} trees cut" + (f" ({'; '.join(notes)})" if notes else "")
                  + "; the game silently skips trees too young for wood, so check next turn "
                  "whether they fell")
    elif kind == "designate":
        candidates = operations.targets.get(action.designation, [])
        matches = [t for t in candidates if action.rect().contains(t['x'], t['z'])]
        if not matches:
            trees = sum(1 for t in client.get_trees() if action.rect().contains(t.position.x, t.position.z))
            hint = f"; it holds {trees} trees: use chop_trees to cut trees" if trees else ""
            raise ValueError(f"no observed {action.designation} target in {action.rect()}{hint}")
        if action.designation == "hunt":
            # Bound the order to the selected, observed animal. Area hunting can
            # accidentally mark predators or faction animals elsewhere in a box.
            if action.rect().cells != 1:
                raise ValueError("hunt must target one observed animal cell (x1=x2,z1=z2)")
            here = client.get_things_at(action.x1, action.z1)
            if not any(t.thing_id == m['id'] for t in here for m in matches):
                raise ValueError("animal moved; choose a fresh target")
        if action.designation == "mine" and mines is not None:
            fresh = [t for t in matches if (t["x"], t["z"]) not in mines.cells]
            if not fresh:
                return (f"unchanged: {target}: all {len(matches)} rock cells there are already marked for mining; "
                        "see 'Mining you ordered' for what became of them")
            client.designate(action.designation, action.rect())
            mines.add(fresh)
            already = len(matches) - len(fresh)
            detail = (f"marked {len(fresh)} rock cells for mining" + (f" ({already} there already were)" if already else "")
                      + "; colonists with Mining on dig them out while time advances")
        else:
            client.designate(action.designation, action.rect())
            detail = "designation submitted; API cannot verify changed count; previously marked/ineligible targets may be unchanged"
    else:
        methods = {
            "place_blueprint": lambda: client.place_blueprint(action.building_def, action.x, action.z, action.rotation, action.stuff),
            "build_wall": lambda: client.build_wall(action.rect(), action.stuff),
            "build_floor": lambda: client.build_floor(action.rect(), action.terrain),
            "build_room": lambda: client.build_room(action.rect(), action.stuff, action.door_side),
        }
        if kind not in methods:
            raise ValueError(f"unsupported action: {kind}")
        if kind == "place_blueprint" and client.construction_at(
                action.building_def, action.x, action.z, action.rotation, stuff=action.stuff):
            return f"unchanged: {target} already built or queued; existing materials retained"
        if kind == "place_blueprint" and action.building_def in ONE_AT_A_TIME and blueprints is not None:
            waiting = [bp for bp, status in blueprints.check(client, forget_finished=False)
                       if bp.def_name == action.building_def and status in ("waiting", "under construction")]
            if waiting:
                bp = waiting[0]
                raise ValueError(f"your {bp.def_name} at ({bp.x},{bp.z}) is still unbuilt; a second one "
                                 "before it stands only doubles the material bill")
        try:
            detail = str(methods[kind]())
        finally:
            # An HTTP failure may follow partial game mutation. Keep every submitted
            # location so next observation reconciles reality, never blindly retry.
            for bp in client.submitted_blueprints:
                blueprints.add(**bp)
            client.begin_snapshot()
        if detail.startswith("unchanged:"):
            return detail + "; target=" + target
        submitted = {(p['def_name'], p['x'], p['z']) for p in client.submitted_blueprints}
        report = blueprints.check(client, forget_finished=False)
        matched = [(bp, status) for bp, status in report if (bp.def_name, bp.x, bp.z) in submitted]
        verified = sum(status != "gone" for _, status in matched)
        if verified != len(submitted):
            raise ValueError(f"construction partially verified: {verified}/{len(submitted)} visible; "
                             "all submitted locations retained for reconciliation; " + detail)
        detail = f"verified {verified} construction orders, not completed buildings; {detail}"
    return target + ": " + detail
