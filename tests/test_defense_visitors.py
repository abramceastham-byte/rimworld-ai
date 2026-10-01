"""Fixes from the 2026-09-24 00:12 gpt-oss run: visitors nobody noticed, walls
built for the Need defenses alert, a plan its toolkit couldn't finish.

Run from the repo root:  python -m unittest tests/test_defense_visitors.py -v
"""

import json
import unittest

from rimagent.construction import BUILDINGS, Rect
from rimagent.defense import defense_lines
from rimagent.modes import Overview, TOOLKITS, plan_mismatch, toolkit_work
from rimagent.rimapi import Enemy, Visitors


class VisitorTests(unittest.TestCase):
    def test_a_visiting_group_is_named_and_marked_friendly(self) -> None:
        group = Visitors(faction_name="Pact of Minbanoulna", relation="Neutral", behaviour="LordJob_VisitColony",
                         names=["Trev", "Unknown"], count=2)
        text = group.summary()
        self.assertIn("2 from Pact of Minbanoulna (neutral): Trev; visiting", text)
        self.assertIn("don't draft against them or attack them", text)


class DefenseTests(unittest.TestCase):
    def test_says_what_counts_and_where_attacks_come_from(self) -> None:
        mechs = [Enemy(id=1, kind="Mech_Militor", x=150, z=128, active=False),
                 Enemy(id=2, kind="Mech_Lancer", x=152, z=130, active=False)]
        lines = defense_lines([("Barricade", Rect(1, 1, 1, 1), "planned")], [(123, 145)], mechs, (120, 138), 1, 3)
        text = "\n".join(lines)
        self.assertIn("Defenses built or planned: 1 Barricade. Colonists armed: 1 of 3.", text)
        self.assertIn("A line of wall doesn't count and protects nothing", text)
        self.assertIn("Doors raiders would come to: (123,145).", text)
        self.assertIn("2 dormant enemies around (151,129), 31 cells east of the base", text)

    def test_sleeping_enemies_nobody_can_locate_are_still_mentioned(self) -> None:
        # RIMAPI's /map/pawns leaves sleeping mechanoids out, so they have no position.
        text = "\n".join(defense_lines([], [], [], (120, 138), 0, 3, dormant=6))
        self.assertIn("6 dormant enemies (see Threats) can't be located", text)
        self.assertNotIn("No enemies on the map", text)

    def test_the_security_tab_is_buildable(self) -> None:
        self.assertEqual(BUILDINGS["TrapSpike"].stuff_count, 45)
        self.assertEqual(BUILDINGS["Turret_MiniTurret"].research, "GunTurrets")
        self.assertIn("defense", TOOLKITS)
        self.assertIn("place_blueprint", TOOLKITS["defense"].actions)
        self.assertEqual(toolkit_work(["defense"]), {"Construction"})


class PlanCheckTests(unittest.TestCase):
    def plan(self, toolkit, *finish):
        return Overview(goal="g", uses="u", toolkits=[toolkit], finish=list(finish))

    def test_a_finish_its_toolkit_cant_make_is_named(self) -> None:
        # Live: wood_mining, done when "stone blocks present in stockpile".
        text = plan_mismatch(self.plan("wood_mining", {"kind": "have_items", "what": "BlocksGranite", "count": 20}))
        self.assertIn('"20 BlocksGranite usable" needs the production toolkit, and you picked wood_mining', text)

    def test_a_finish_its_toolkit_makes_passes(self) -> None:
        self.assertIsNone(plan_mismatch(self.plan("production", {"kind": "have_items", "what": "BlocksGranite"})))
        self.assertIsNone(plan_mismatch(self.plan("construction", {"kind": "room_closed", "what": "bedroom"},
                                                  {"kind": "beds", "count": 3})))
        self.assertIsNone(plan_mismatch(self.plan("defense", {"kind": "built", "what": "Barricade", "count": 4})))
        self.assertIsNotNone(plan_mismatch(self.plan("construction", {"kind": "zone_cells", "what": "stockpile",
                                                                      "count": 20})))

    def test_the_overview_is_asked_again_once_then_taken(self) -> None:
        from rimagent.agent import request_overview

        class Log:
            def __init__(self): self.failures = []
            def log_model_failure(self, record): self.failures.append(record)

        blocks = [{"kind": "have_items", "what": "BlocksGranite", "count": 20}]
        bad = json.dumps({"goal": "g", "uses": "u", "toolkits": ["wood_mining"], "finish": blocks})
        good = json.dumps({"goal": "g", "uses": "u", "toolkits": ["production"], "finish": blocks})
        prompts = []

        def model(prompt):
            prompts.append(prompt)
            return bad if len(prompts) == 1 else good
        log = Log()
        overview, _ = request_overview(model, "PROMPT", log, 0, frozenset({"BlocksGranite"}))
        self.assertEqual(overview.toolkits, ["production"])
        self.assertIn("It can't be finished as it stands", prompts[1])
        self.assertEqual(log.failures[0]["failure_type"], "overview_mismatch")

        prompts.clear()
        overview, _ = request_overview(lambda p: prompts.append(p) or bad, "PROMPT", Log(), 0,
                                       frozenset({"BlocksGranite"}))
        self.assertEqual(overview.toolkits, ["wood_mining"])  # asked once, then taken as it is


class FinishTests(unittest.TestCase):
    def test_checks_are_measured_and_all_met_ends_the_plan(self) -> None:
        from rimagent.finish import FinishCheck, FinishState, finish_progress
        checks = [FinishCheck(kind="room_closed", what="storage"), FinishCheck(kind="built", what="Shelf", count=3),
                  FinishCheck(kind="zone_cells", what="stockpile", count=10)]
        state = FinishState(rooms_closed={"storage": 1}, built={"Shelf": 2}, zone_cells={"stockpile": 16})
        lines, met = finish_progress(checks, state)
        self.assertEqual(lines, ["- 1 storage room closed: MET", "- 3 Shelf built: 2 of 3",
                                 "- 10 cells of stockpile zone: MET"])
        self.assertFalse(met)
        state.built["Shelf"] = 3
        self.assertTrue(finish_progress(checks, state)[1])

    def test_a_check_must_name_something_real(self) -> None:
        from pydantic import ValidationError
        from rimagent.finish import FinishCheck
        with self.assertRaisesRegex(ValidationError, "room purpose"):
            FinishCheck(kind="room_closed", what="fortress")
        with self.assertRaisesRegex(ValidationError, "building def"):
            FinishCheck(kind="built", what="WoodenSpear")

    def test_steps_are_marked_done_and_current(self) -> None:
        from rimagent.agent import track_steps
        from rimagent.memory import CurrentPlan, PlanStep
        plan = CurrentPlan(objective="o", steps=[PlanStep(description=d, status="pending") for d in "abc"])
        track_steps(plan, 2, [1])
        self.assertEqual([s.status for s in plan.steps], ["completed", "in_progress", "pending"])
        track_steps(plan, 3, [])
        self.assertEqual([s.status for s in plan.steps], ["completed", "in_progress", "in_progress"])
        from rimagent.modes import plan_block
        text = "\n".join(plan_block(plan))
        self.assertIn("1. [done] a", text)
        self.assertIn("3. [now] c", text)


if __name__ == "__main__":
    unittest.main()
