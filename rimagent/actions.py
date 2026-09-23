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


class Wait(BaseAction):
    action: Literal["wait"]


class Pause(BaseAction):
    action: Literal["pause"]


class Resume(BaseAction):
    action: Literal["resume"]


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
    """A straight run of walls, so a room is four actions instead of a dozen."""

    action: Literal["build_wall"]
    stuff: str


class BuildFloor(RectAction):
    action: Literal["build_floor"]
    terrain: str  # floor def name, e.g. "WoodPlankFloor"


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
        Wait, Pause, Resume, EnableWork, DisableWork, SetBill, SetResearch,
        CreateGrowingZone, CreateStockpile, PlaceBlueprint, Designate,
        BuildWall, BuildFloor,
        AllowItems, ChopTrees,
    ],
    Field(discriminator="action"),
]


class Turn(BaseModel):
    """One reply: up to MAX_ACTIONS actions, carried out in order.

    Several actions at once let the model lay out a room or set up a colonist
    in one go, instead of one cell per decision. Pausing first (see TimeMode)
    means no game time passes in between.
    """

    model_config = ConfigDict(extra="forbid")

    actions: list[Action] = Field(min_length=1, max_length=MAX_ACTIONS)
    memory_update: MemoryUpdate | None = None
