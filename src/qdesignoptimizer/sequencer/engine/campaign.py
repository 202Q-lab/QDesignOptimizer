"""Campaigns: lab-log folders that each answer one question.

A campaign is a folder under ``<project>/campaigns/`` with a
``README.md`` saying the question, what was tried and what was
concluded (*tune up the whole chip*, *a Purcell filter*, *mesh
sensitivity*). A campaign inside another is a sub-campaign; it is just
a campaign, nested::

    <project>/campaigns/
      tuneup_chip/
        README.md
        design_variables.json      the campaign's own copy of the store
        q1_readout/                a sub-campaign
          README.md                (no store: it uses tuneup_chip's)

A campaign's key is its path below ``campaigns/``
(``tuneup_chip/q1_readout``): lower-case words joined by ``_``, no
dates (the README and the studies carry those).

A study queued in a campaign says so in its ``job.yaml``
(``campaign: tuneup_chip/q1_readout``); its id and folder stay as
they are, so solver paths stay short. Such a study reads and writes the
campaign's store: the nearest ``design_variables.json`` from its
campaign up, the project's own when no campaign on the way has one. A
new campaign starts with its own copy of the store it would otherwise
use, so a campaign changes nothing outside itself until its values are
promoted (``promote``) one level up.

The backend says which file that is (``store_for``) and reads and
writes it; the engine never reads a store itself.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

CAMPAIGN_DIR = "campaigns"
README = "README.md"

_PART = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class CampaignError(ValueError):
    """A campaign key or folder cannot be used."""


@dataclass(frozen=True)
class CampaignInfo:
    """One campaign, as the folder has it.

    Attributes:
        key: Its path below ``campaigns/`` (``tuneup_chip/q1_readout``).
        title: The README's first heading, or the key.
        folder: The campaign's folder.
        own_store: Whether it holds its own copy of the store.
    """

    key: str
    title: str
    folder: Path
    own_store: bool

    @property
    def parent(self) -> str:
        """The enclosing campaign's key, "" for a top-level one."""
        return self.key.rpartition("/")[0]

    @property
    def depth(self) -> int:
        """0 for a top-level campaign, 1 for its sub-campaigns, ..."""
        return self.key.count("/")


def check_key(key: str) -> str:
    """The key, normalised (``\\`` to ``/``, no outer slashes).

    Raises:
        CampaignError: If a part is not lower-case words joined by ``_``.
    """
    key = str(key).replace("\\", "/").strip("/")
    parts = key.split("/") if key else []
    if not parts:
        raise CampaignError("a campaign needs a name")
    for part in parts:
        if not _PART.match(part):
            raise CampaignError(
                f"campaign name {part!r} must be lower-case letters, digits "
                "and _ (for example tuneup_chip or mesh_sensitivity)"
            )
    return key


def root_of(project_root: Path) -> Path:
    """``<project>/campaigns``."""
    return Path(project_root) / CAMPAIGN_DIR


def folder_of(project_root: Path, key: str) -> Path:
    """The folder of campaign ``key`` (it need not exist)."""
    return root_of(project_root).joinpath(*check_key(key).split("/"))


def exists(project_root: Path, key: str) -> bool:
    """Whether campaign ``key`` is there (a folder with a README)."""
    try:
        return (folder_of(project_root, key) / README).is_file()
    except CampaignError:
        return False


def problems_of(project_root: Path, key: str) -> list[str]:
    """What is wrong with a study's ``campaign:``; [] when it can run."""
    try:
        check_key(key)
    except CampaignError as exc:
        return [str(exc)]
    if not exists(project_root, key):
        return [
            f"campaign {key!r} does not exist (no " f"{CAMPAIGN_DIR}/{key}/{README})"
        ]
    return []


def campaigns(project_root: Path, store_name: str = "") -> list[CampaignInfo]:
    """Every campaign of a project, parents before their sub-campaigns,
    by key. ``own_store`` says whether it holds a ``store_name`` (False
    without one)."""
    top = root_of(project_root)
    if not top.is_dir():
        return []
    found = []
    for readme in sorted(top.rglob(README)):
        folder = readme.parent
        if folder == top:
            continue
        key = folder.relative_to(top).as_posix()
        try:
            check_key(key)
        except CampaignError:
            continue  # a stray folder (Results, .git); not a campaign
        found.append(
            CampaignInfo(
                key=key,
                title=_title(readme) or key,
                folder=folder,
                own_store=bool(store_name) and (folder / store_name).is_file(),
            )
        )
    return sorted(found, key=lambda c: c.key.split("/"))


def store_file(project_root: Path, key: str, store_name: str, default: Path) -> Path:
    """The store a study in campaign ``key`` reads and writes: the
    nearest ``store_name`` from the campaign up to ``campaigns/``,
    ``default`` (the project's) when none has one. ``key`` "" is the
    project itself."""
    if not key:
        return Path(default)
    top = root_of(project_root).resolve()
    folder = folder_of(project_root, key).resolve()
    while folder != top and top in folder.parents:
        candidate = folder / store_name
        if candidate.is_file():
            return candidate
        folder = folder.parent
    return Path(default)


def create(
    project_root: Path,
    key: str,
    question: str = "",
    copy_of: Path | None = None,
) -> Path:
    """Make campaign ``key``: its folder and a README to fill in.

    A sub-campaign's parent must exist already. ``copy_of`` is the store
    it would otherwise use (its parent's, or the project's): given, the
    campaign gets its own copy of it, with that store's write record.

    Raises:
        CampaignError: For a bad key, a missing parent, or one that
            exists.
    """
    key = check_key(key)
    parent = key.rpartition("/")[0]
    if parent and not exists(project_root, parent):
        raise CampaignError(
            f"make campaign {parent!r} first: {key!r} is a sub-campaign of it"
        )
    folder = folder_of(project_root, key)
    if (folder / README).exists():
        raise CampaignError(f"campaign {key!r} exists already")
    folder.mkdir(parents=True, exist_ok=True)
    title = key.rpartition("/")[2].replace("_", " ")
    (folder / README).write_text(
        f"# {title[:1].upper() + title[1:]}\n\n"
        f"**Question.** {question or '(what this campaign answers)'}\n\n"
        "## Log\n\n"
        "- (date: what was run, under which study ids, and what it showed)\n\n"
        "## Conclusion\n\n"
        "(the answer, and whether its values were promoted)\n",
        encoding="utf-8",
    )
    if copy_of is not None and Path(copy_of).is_file():
        copy_of = Path(copy_of)
        shutil.copyfile(copy_of, folder / copy_of.name)
        meta = _meta_of(copy_of)
        if meta.is_file():
            shutil.copyfile(meta, _meta_of(folder / copy_of.name))
    return folder


def _meta_of(store: Path) -> Path:
    return store.with_name(store.stem + ".meta.json")


def _title(readme: Path) -> str:
    try:
        for line in readme.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return ""
