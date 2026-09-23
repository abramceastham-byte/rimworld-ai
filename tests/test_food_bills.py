import unittest
from unittest.mock import Mock, call

from pydantic import ValidationError

from rimagent.actions import SetBill
from rimagent.agent import describe_work_table
from rimagent.rimapi import RimApiClient, WorkTable


RECIPES = [
    {
        "def_name": "CookMealSimple",
        "label": "cook simple meal",
        "description": "Cook a simple meal.",
        "work_amount": 300,
        "work_skill": "Cooking",
        # As RIMAPI 1.10 sends them (extra fields, kept by Recipe).
        "ingredients": [{"filter_label": "raw food", "count": 10.0}],
        "products": [{"thing_def": "MealSimple", "count": 1}],
    }
]


class DecisionTests(unittest.TestCase):
    def test_set_bill_requires_all_fields(self) -> None:
        with self.assertRaises(ValidationError):
            SetBill(action="set_bill", building_id=12, recipe_def_name="CookMealSimple", reason="x")

    def test_set_bill_accepts_safe_target_count(self) -> None:
        decision = SetBill(
            action="set_bill",
            building_id=12,
            recipe_def_name="CookMealSimple",
            target_count=20,
            reason="Keep meals stocked.",
        )
        self.assertEqual(decision.target_count, 20)


class BillClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = RimApiClient(base_url="http://example.invalid")

    def tearDown(self) -> None:
        self.client.close()

    def test_set_target_bill_creates_when_recipe_has_no_bill(self) -> None:
        request = Mock(side_effect=[RECIPES, [], {"load_id": 44}])
        self.client._request = request

        self.client.set_target_bill(12, "CookMealSimple", 20)

        self.assertEqual(
            request.call_args_list,
            [
                call(
                    "GET",
                    "/api/v1/buildings/recipes",
                    params={"building_id": 12, "only_researched": True},
                ),
                call(
                    "GET",
                    "/api/v1/buildings/bills",
                    params={"building_id": 12},
                ),
                call(
                    "POST",
                    "/api/v1/buildings/bills/add",
                    params={"building_id": 12},
                    json={
                        "recipe_def_name": "CookMealSimple",
                        "repeat_mode": "TargetCount",
                        "target_count": 20,
                    },
                ),
            ],
        )

    def test_set_target_bill_updates_and_resumes_existing_bill(self) -> None:
        request = Mock(
            side_effect=[
                RECIPES,
                [
                    {
                        "load_id": 44,
                        "recipe_def_name": "CookMealSimple",
                        "repeat_mode": "RepeatCount",
                        "repeat_count": 5,
                        "suspended": True,
                    }
                ],
                None,
                None,
            ]
        )
        self.client._request = request

        self.client.set_target_bill(12, "CookMealSimple", 30)

        params = {"building_id": 12, "bill_id": 44}
        request.assert_has_calls(
            [
                call(
                    "PUT",
                    "/api/v1/buildings/bill/update",
                    params=params,
                    json={"repeat_mode": "TargetCount", "target_count": 30},
                ),
                call(
                    "PUT",
                    "/api/v1/buildings/bill/suspend",
                    params=params,
                    json={"suspended": False},
                ),
            ]
        )

    def test_set_target_bill_rejects_recipe_before_writing(self) -> None:
        request = Mock(return_value=RECIPES)
        self.client._request = request

        with self.assertRaisesRegex(ValueError, "is not available"):
            self.client.set_target_bill(12, "InventedRecipe", 20)

        self.assertEqual(request.call_count, 1)
        self.assertTrue(request.call_args.kwargs['params']['only_researched'])

    def test_observation_requests_only_unlocked_recipes(self) -> None:
        request = Mock(side_effect=[[
            {'id': 12, 'thing_def': 'ElectricStove', 'label': 'stove', 'position': {'x': 1, 'z': 1}}
        ], RECIPES, []])
        self.client._request = request
        tables = self.client.get_work_tables()
        self.assertEqual([r.def_name for r in tables[0].recipes], ['CookMealSimple'])
        self.assertEqual(request.call_args_list[1], call('GET', '/api/v1/buildings/recipes',
                         params={'building_id': 12, 'only_researched': True}))


class PromptDescriptionTests(unittest.TestCase):
    def test_work_table_description_contains_exact_action_values(self) -> None:
        table = WorkTable.model_validate(
            {
                "id": 12,
                "thing_def": "ElectricStove",
                "label": "electric stove",
                "position": {"x": 20, "y": 0, "z": 30},
                "recipes": RECIPES,
                "bills": [],
            }
        )

        description = "\n".join(describe_work_table(table))

        self.assertIn("id 12", description)
        # Ingredients from RIMAPI; the product is left out when the def name says it.
        self.assertIn("CookMealSimple: 10 raw food", description)
        self.assertNotIn("-> 1 MealSimple", description)
        self.assertIn("use Cooking", description)


if __name__ == "__main__":
    unittest.main()
