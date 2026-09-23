"""What the model is allowed to reply: a turn of actions, one model per action.

Each action is its own model with exactly the fields it needs, joined into a
tagged union on "action". The JSON schema handed to Ollama then says, for
example, that create_growing_zone must have plant and x1/z1/x2/z2 and cannot
have x/z/rotation/stuff. With one flat model (every field optional for every
action) models kept sending a blueprint's x/z for a zone, and only pydantic
caught it, after a wasted call.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from rimagent.construction import Rect
from rimagent.memory import MemoryUpdate

MAX_ACTIONS = 5


class BaseAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "action" must be the first field of every action. Constrained decoding
    # writes properties in schema order and has to choose a branch of the union
    # at the first one; if "reason" came first the branch was picked before the
    # action name existed, and the action was then forced to whatever that
    # branch was (a whole run of place_blueprint with unrelated reasons).
    action: str
    reason: str = Field(min_length=1, max_length=300)


class RectAction(BaseAction):
    """An action over a rectangle of cells, from (x1,z1) to (x2,z2)."""

    x1: int = Field(ge=0)
    z1: int = Field(ge=0)
    x2: int = Field(ge=0)
    z2: int = Field(ge=0)

    def rect(self) -> Rect:
        return Rect.from_corners(self.x1, self.z1, self.x2, self.z2)


class EnableWork(BaseAction):
    action: Literal["enable_work"]
    colonist: str
    work_type: str


class DisableWork(BaseAction):
    action: Literal["disable_work"]
    colonist: str
    work_type: str


class SetBill(BaseAction):
    action: Literal["set_bill"]
    building_id: int = Field(gt=0)
    recipe_def_name: str
    target_count: int = Field(ge=1, le=500)


class CreateGrowingZone(RectAction):
    action: Literal["create_growing_zone"]
    plant: str


class CreateStockpile(RectAction):
    action: Literal["create_stockpile"]


class Designate(RectAction):
    action: Literal["designate"]
    designation: Literal["mine", "harvest", "hunt"]


class AllowItems(RectAction):
    action: Literal["allow_items"]


class ChopTrees(RectAction):
    action: Literal["chop_trees"]


class BuildWall(RectAction):
    """A straight run of walls; build_room is the one-action way to a whole room."""

    action: Literal["build_wall"]
    stuff: str


class BuildFloor(RectAction):
    action: Literal["build_floor"]
    terrain: str  # floor def name, e.g. "WoodPlankFloor"


class BuildRoom(RectAction):
    """A closed ring of walls around the outer rectangle, with one centred door
    in the chosen side (see RimApiClient.build_room)."""

    action: Literal["build_room"]
    stuff: str
    door_side: Literal["north", "east", "south", "west"]


class SetResearch(BaseAction):
    action: Literal["set_research"]
    project: str  # def name from the available list, e.g. "Brewing"


class PlaceBlueprint(BaseAction):
    action: Literal["place_blueprint"]
    building_def: str
    x: int = Field(ge=0)
    z: int = Field(ge=0)
    rotation: int = Field(default=0, ge=0, le=3)
    stuff: str | None = None  # material, when the building needs one


Action = Annotated[
    Union[
        EnableWork, DisableWork, SetBill, SetResearch,
        CreateGrowingZone, CreateStockpile, PlaceBlueprint, Designate,
        BuildWall, BuildFloor, BuildRoom,
        AllowItems, ChopTrees,
    ],
    Field(discriminator="action"),
]


class Turn(BaseModel):
    """One reply: up to MAX_ACTIONS orders, carried out in order while the game
    is paused, then a choice of whether game time runs before the next turn.

    Time is chosen every turn, so no pause state can be forgotten. It comes
    after the orders: constrained decoding writes fields in schema order, and
    the model should choose it knowing what it just ordered.
    """

    model_config = ConfigDict(extra="forbid")

    actions: list[Action] = Field(max_length=MAX_ACTIONS)
    time: Literal["advance", "hold"]
    memory_update: MemoryUpdate | None = None
