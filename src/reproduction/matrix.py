"""Selected mask-search and downstream matrices for the final method."""

from dataclasses import dataclass
from typing import Literal
from .config import load_contract

Dataset = Literal["wmdpall", "muse-news", "muse-books"]
Updater = Literal["gd", "npo"]


@dataclass(frozen=True, slots=True)
class Stage1Cell:
    dataset: Dataset
    support: float
    selector: str
    seed: int = 3


@dataclass(frozen=True, slots=True)
class Stage2Cell:
    dataset: Dataset
    support: float
    updater: Updater
    selector: str
    seed: int
    mask_seed: int = 3

    def __post_init__(self):
        if self.selector not in selectors_for_updater(self.updater):
            raise ValueError(f"Invalid selector for {self.updater}: {self.selector}")

    @property
    def recipe(self):
        return load_contract()["datasets"][self.dataset]["stage2"]["recipes"][
            self.updater
        ]


@dataclass(frozen=True, slots=True)
class MaskKey:
    dataset: Dataset
    support: float
    selector: str
    seed: int = 3


def selectors_for_updater(updater):
    if updater not in ("gd", "npo"):
        raise ValueError(f"Unknown updater: {updater}")
    return ("gec", "wagle", f"tcus-{updater}")


def artifact_selector_name(selector):
    if selector == "gec":
        return "normdiff"
    if selector not in ("wagle", "tcus-gd", "tcus-npo"):
        raise ValueError(f"Unknown selector: {selector}")
    return selector


def _supports(dataset, budgets):
    contract = load_contract()
    return (
        (contract["headline_supports"][dataset],)
        if budgets == "headline"
        else contract["support_fracs"]
    )


def stage1_cells():
    from .settings import current

    stage = current()["stage1"]
    return tuple(
        Stage1Cell(dataset, support, selector, seed)
        for dataset in stage["datasets"]
        for seed in stage["seeds"]
        for support in _supports(dataset, stage["budgets"])
        for selector in stage["selectors"]
    )


def stage2_cells(scope=None):
    from .settings import current

    stage = current()["stage2"]
    budgets = stage["budgets"] if scope is None else scope
    if budgets == "full":
        budgets = "all"
    if budgets not in ("all", "headline"):
        raise ValueError(f"Unknown budget selection: {budgets}")
    return tuple(
        Stage2Cell(
            dataset,
            support,
            updater,
            selector,
            seed,
            seed if stage["mask_seed"] == "match" else stage["mask_seed"],
        )
        for dataset in stage["datasets"]
        for seed in stage["seeds"]
        for support in _supports(dataset, budgets)
        for updater in stage["updaters"]
        for selector in selectors_for_updater(updater)
        if selector.split("-")[0] in stage["selectors"]
    )


def mask_key(cell):
    return MaskKey(
        cell.dataset,
        cell.support,
        cell.selector,
        cell.seed if isinstance(cell, Stage1Cell) else cell.mask_seed,
    )
