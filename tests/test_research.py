"""Research state and starting a project.

Run from the repo root:  python -m unittest tests/test_research.py -v
"""

import unittest

from rimagent.agent import describe_research
from rimagent.rimapi import RimApiClient

# The placeholder /research/progress returns when nothing is being researched.
# Its bench flag is always false, which is why has_bench cannot come from here.
NOTHING = {"name": "none", "label": "None", "progress": 0,
           "player_has_any_appropriate_research_bench": False}
BREWING = {"name": "Brewing", "label": "beer brewing", "research_points": 400,
           "can_start_now": True, "is_finished": False,
           "player_has_any_appropriate_research_bench": True}
BATTERIES = {"name": "Batteries", "label": "batteries", "research_points": 200,
             "can_start_now": True, "is_finished": False,
             "player_has_any_appropriate_research_bench": True}
ELECTRICITY = {"name": "Electricity", "label": "electricity", "is_finished": True,
               "can_start_now": False, "player_has_any_appropriate_research_bench": True}
FABRICATION = {"name": "Fabrication", "label": "fabrication", "can_start_now": False,
               "is_finished": False, "prerequisites": ["Machining"],
               "player_has_any_appropriate_research_bench": False}


class FakeResearch:
    def __init__(self, progress=NOTHING, projects=(BATTERIES, BREWING, ELECTRICITY, FABRICATION)):
        self.progress, self.projects, self.calls = progress, list(projects), []

    def __call__(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if path == "/api/v1/research/progress":
            return self.progress
        if path == "/api/v1/research/tree":
            return {"projects": self.projects}
        return None


class ResearchTests(unittest.TestCase):
    def client(self, fake: FakeResearch) -> RimApiClient:
        client = RimApiClient(base_url="http://example.invalid")
        client._request = fake
        self.addCleanup(client.close)
        return client

    def test_a_bench_is_found_even_when_nothing_is_being_researched(self) -> None:
        state = self.client(FakeResearch()).get_research()
        self.assertTrue(state.has_bench)          # from the tree, not the placeholder
        self.assertIsNone(state.current)
        self.assertEqual([p.name for p in state.available], ["Batteries", "Brewing"])  # cheapest first

    def test_no_bench_at_all(self) -> None:
        no_bench = [{**p, "player_has_any_appropriate_research_bench": False,
                     "can_start_now": False} for p in (BATTERIES, BREWING)]
        self.assertFalse(self.client(FakeResearch(projects=no_bench)).get_research().has_bench)

    def test_current_project(self) -> None:
        working = {"name": "Brewing", "label": "beer brewing", "progress_percent": 40.0,
                   "player_has_any_appropriate_research_bench": True}
        state = self.client(FakeResearch(progress=working)).get_research()
        self.assertEqual(state.current.label, "beer brewing")
        self.assertIn("Researching beer brewing (40% done).", describe_research(state))

    def test_starting_a_project(self) -> None:
        fake = FakeResearch()
        self.assertEqual(self.client(fake).set_research("brewing"), "beer brewing")  # case-insensitive
        (method, path, kwargs), = [c for c in fake.calls if c[1] == "/api/v1/research/target"]
        self.assertEqual((method, kwargs["params"]), ("POST", {"name": "Brewing", "force": False}))

    def test_each_refusal_says_the_real_reason(self) -> None:
        client = self.client(FakeResearch())
        with self.assertRaisesRegex(ValueError, "not a research project"):
            client.set_research("Agriculture")          # a name the model made up
        with self.assertRaisesRegex(ValueError, "already researched"):
            client.set_research("Electricity")
        with self.assertRaisesRegex(ValueError, "needs a research bench"):
            client.set_research("Fabrication")          # needs a better bench

    def test_prerequisites_are_named(self) -> None:
        needs_prereq = {**FABRICATION, "player_has_any_appropriate_research_bench": True}
        client = self.client(FakeResearch(projects=(BREWING, needs_prereq)))
        with self.assertRaisesRegex(ValueError, "needs Machining first"):
            client.set_research("Fabrication")

    def test_no_bench_is_reported_in_the_prompt(self) -> None:
        no_bench = [{**BREWING, "player_has_any_appropriate_research_bench": False,
                     "can_start_now": False}]
        state = self.client(FakeResearch(projects=no_bench)).get_research()
        self.assertIn("No research bench is built", " ".join(describe_research(state)))


if __name__ == "__main__":
    unittest.main()
