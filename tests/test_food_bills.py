import unittest
from unittest.mock import Mock, call

from pydantic import ValidationError

from rimagent.agent import Decision, describe_work_table
from rimagent.rimapi import RimApiClient, WorkTable


RECIPES = [
    {
        "def_name": "CookMealSimple",
        "label": "cook simple meal",
        "description": "Cook a simple meal.",
        "work_amount": 300,
        "work_skill": "Cooking",
    }
]


class DecisionTests(unittest.TestCase):
    def test_set_bill_requires_all_fields(self) -> None:
        with self.assertRaises(ValidationError):
            Decision(action="set_bill", building_id=12, recipe_def_name="CookMealSimple")

    def test_set_bill_accepts_safe_target_count(self) -> None:
        decision = Decision(
            action="set_bill",
            building_id=12,
            recipe_def_name="CookMealSimple",
            target_count=20,
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
                    params={"building_id": 12},
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
        self.assertIn("CookMealSimple", description)
        self.assertIn("cook simple meal", description)


if __name__ == "__main__":
    unittest.main()
