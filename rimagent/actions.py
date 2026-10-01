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
from rimagent.layout import AreaPurpose
from rimagent.memory import MemoryUpdate

MAX_ACTIONS = 10


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


class AllowItems(BaseAction):
    """Allow forbidden items anywhere on the map: all of them, all food, or one def.

    It took a rectangle, and a model spent four turns allowing one scattered
    pile at a time (64 logs on one cell, 4 meals 60 cells out) while nobody
    had work. Walking near the dormant mechanoids doesn't wake them.
    """

    action: Literal["allow_items"]
    what: str = "all"  # "all", "Food" (meals and raw food), or an item def name


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
    # What the room is for; tracked so later prompts say what it still needs.
    purpose: Literal["bedroom", "barracks", "kitchen", "dining", "workshop", "storage", "hospital", "other"]


class ConnectPower(BaseAction):
    """Link two buildings with conduits; code chooses the route (see rimagent.power)."""

    action: Literal["connect_power"]
    from_x: int = Field(ge=0)  # any cell of the power source (generator, battery) or its grid
    from_z: int = Field(ge=0)
    to_x: int = Field(ge=0)    # any cell of what needs power (lamp, stove, ...)
    to_z: int = Field(ge=0)


class PlanArea(RectAction):
    """Reserve where a kind of building goes (see rimagent.layout). Builds nothing."""

    action: Literal["plan_area"]
    purpose: AreaPurpose


class Draft(BaseAction):
    action: Literal["draft"]
    colonist: str
    drafted: bool  # true to draft for combat, false to release back to work


class Move(BaseAction):
    """Send a drafted colonist to a cell."""

    action: Literal["move"]
    colonist: str
    x: int = Field(ge=0)
    z: int = Field(ge=0)


class Attack(BaseAction):
    """A drafted colonist attacks one enemy: shoots with a gun, else melee."""

    action: Literal["attack"]
    colonist: str
    target: int  # enemy id from the threat list


class Equip(BaseAction):
    """Pick up a weapon (or put on apparel) lying on the map."""

    action: Literal["equip"]
    colonist: str
    item_id: int


class Tend(BaseAction):
    action: Literal["tend"]
    patient: str
    doctor: str | None = None  # None: any capable doctor


class BedRest(BaseAction):
    action: Literal["bed_rest"]
    patient: str


class Rescue(BaseAction):
    """Carry a downed colonist to a bed."""

    action: Literal["rescue"]
    colonist: str
    patient: str


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
        ConnectPower, PlanArea, Draft, Move, Attack, Equip, Tend, BedRest, Rescue,
    ],
    Field(discriminator="action"),
]


class Turn(BaseModel):
    """One reply: up to MAX_ACTIONS orders, each carried out in order (a failure
    doesn't stop the rest) while the game
    is paused, then a choice of whether game time runs before the next turn.

    Time is chosen every turn, so no pause state can be forgotten. It comes
    after the orders: constrained decoding writes fields in schema order, and
    the model should choose it knowing what it just ordered.
    """

    model_config = ConfigDict(extra="forbid")

    actions: list[Action] = Field(max_length=MAX_ACTIONS)
    time: Literal["advance", "hold"]
    memory_update: MemoryUpdate | None = None
