"""Execute one order, distinguish unchanged, verified and submitted outcomes."""

import json

from rimagent.choices import action_target, incapable_work
from rimagent.rimapi import RimApiError


def execute(client, action, blueprints, zones, operations, learned_incapable=None):
    """Carry out one order against freshly read state, and say what it did.

    Returns "unchanged: ..." when the game already matches, or what was
    verified or merely submitted. Raises ValueError/RimApiError on failure.
    learned_incapable collects work RIMAPI refused, for later turns' choices.
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
            return f"unchanged: {target} already {'enabled' if wanted else 'disabled'}"
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
            return f"{target}: {detail}"
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
        count = client.chop_trees(action.rect())
        detail = f"submitted harvest requests at {count} tree cells; changed count/readiness unavailable; no wood obtained yet"
    elif kind == "designate":
        candidates = operations.targets.get(action.designation, [])
        matches = [t for t in candidates if action.rect().contains(t['x'], t['z'])]
        if not matches:
            raise ValueError(f"no observed {action.designation} candidate in {action.rect()}")
        if action.designation == "hunt":
            # Bound the order to the selected, observed animal. Area hunting can
            # accidentally mark predators or faction animals elsewhere in a box.
            if action.rect().cells != 1:
                raise ValueError("hunt must target one observed animal cell (x1=x2,z1=z2)")
            here = client.get_things_at(action.x1, action.z1)
            if not any(t.thing_id == m['id'] for t in here for m in matches):
                raise ValueError("animal moved; choose a fresh target")
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
