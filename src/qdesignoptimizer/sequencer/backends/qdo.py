"""The QDO backend: a queued study is a folder of files QDO runs from.

::

    job.yaml                 engine: the queue entry (which study, its
                             campaign, run settings, checks, save)
    ministudy.yaml           the MiniStudy, as its constructor call
    opt_targets.yaml         the OptTargets (QDO helper calls or explicit)
    parameter_targets.yaml   optional: only the targets you set here
    design_variables.json    optional: only the variables you set here

A study exists only as such a folder. It is made from the project's
templates (see ``qdo_templates``): one MiniStudy template
(``<design>/ministudies/<name>.yaml``, the MiniStudy alone), filled in,
and every opt target of the project (``<design>/opt_targets/*.yaml``)
whose modes are all modes of that MiniStudy -- for a ``capacitance``
target, whose Ansys conductors belong to the components of one of its
chained capacitance studies (``measures_capacitance``). The files are what runs,
so an edit there is the study. The two optional files hold only what
you set; when the study starts, everything else is pulled from the
project -- after every study above it has written back -- and the
completed folder records the full set that ran.

**Which targets start switched on.** Every fitting target is written
into the study; one the study cannot produce is written with
``active: false`` (kept, so it is one flag to switch on in a copy of the
study, and left out when the OptTargets are built). A target is tuned in
one kind of simulation -- its ``simulation:`` field, else by its
parameter: ``kappa``, ``charge_line_limited_t1`` and ``capacitance`` in
a capacitance study, everything else in an eigenmode solve -- and starts
on when the MiniStudy measures its parameter in that kind
(``study_parameters``): the eigenmode solve gives each mode's frequency
and linewidth and the nonlinearities, each chained decay study the one
parameter of its type for its mode (and wins over the eigenmode value,
as in QDO). Of two targets that tune one design variable or chase one
parameter, only the first starts on, and the study check refuses two
switched on. A target the project itself marks ``active: false`` stays
off.

**Converged** is judged per target, by its ``tolerance:`` (relative, or
``{abs: ...}``) against the last iteration; a project's ``converged``
hook, when it has one, judges the targets that set none.

Solver settings (passes, ``delta_f``, mesh) are written in the
MiniStudy, and the sequencer has no setting of its own that overrides
them. What a project's ``solve`` does with them is the project's.

What a step writes back: the design variables its OptTargets tune and
the optimizer moved, compared with the project's file at the end. A
variable you set in the folder that no target tunes is used for this
study only and never written back.

Nothing here imports QDO or qiskit-metal at module scope; the project
hooks and ``qdo_files`` do, when a study is generated, checked or run.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

import yaml

from ..engine.backend import (
    Catalog,
    CheckResult,
    FileSpec,
    StepResult,
    SubjectSpec,
    TaskSpec,
    Words,
)
from ..engine.campaign import load_campaign_dir
from ..engine.job import dump_yaml, is_custom, numbers
from . import qdo_record
from ._store import read_json, record_writes, sources_text, update_json
from .qdo_templates import (
    MINISTUDY_DIR,
    TARGETS_DIR,
    Template,
    TemplateError,
    discover_templates,
    load_targets,
    render,
    template_files,
    write_ministudy,
    write_targets,
)

log = logging.getLogger(__name__)

WORDS = Words(
    task="study",
    campaign="campaign",
    subject="subsystem",
    store="design variables",
    live="solve",
)

CAMPAIGN_DIR = "campaigns"
MINISTUDY, TARGETS = "ministudy.yaml", "opt_targets.yaml"
PARAMS, VARIABLES = "parameter_targets.yaml", "design_variables.json"

_MINISTUDY_HEADER = """\
MiniStudy for {task}, made {when} from
  {by}
The study runs from this file, not from the template: edit freely.
The solver settings (nbr_passes, delta_f, mesh) are set here; the
queue entry (job.yaml) has none.
The template's "=..." values (a capacitance study's mode_freq_GHz, for
one) were filled in from the project when this file was written."""

_TARGETS_HEADER = """\
OptTargets for {task}, generated {when}:
every opt target of the project whose modes are all modes of this
study's MiniStudy. One with
"active: false" is kept here and not used; switch it on to use it.
Per target, optional: tolerance (relative, or {{abs: ...}}) -- how close
the last iteration must come for the "converged" check.
Two forms:
  helper: + args:  calls that QDO helper; the formula stays QDO's
  explicit fields  prop_to is an expression of p['<param>'] and
                   v['<design variable>'] (+ - * / **, abs, sqrt, exp,
                   log, log10, min, max, pi, e)"""

_PARAMS_HEADER = """\
Parameter targets for this study only. Leave a target out and the
project's value (parameter_targets.py) is used. When the study starts
this file is filled in with every target that ran."""


class QDOProject(Protocol):
    """How a design project plugs in: its adapter (``sequencer.py``).

    The adapter says HOW the project solves. WHAT can be queued is not
    listed here: it is the project's templates, in ``<root>/ministudies``
    and ``<root>/opt_targets`` (or the folders ``ministudy_dir`` and
    ``opt_targets_dir`` name), and its campaigns, in
    ``<root>/campaigns`` (or ``campaign_dir``).
    """

    #: The design folder; ``out/`` lives under it.
    root: Path
    #: The design variables file: the store.
    design_variables_file: Path

    def settings_keys(self) -> set[str]:
        """What ``job.yaml``'s ``settings`` may hold: how the study is
        run (iterations, retries), not what the solver is told -- that
        is the MiniStudy's."""

    def default_settings(self, task: TaskSpec) -> dict:
        """The settings a new study of this template starts with."""

    def parameter_targets(self) -> dict[str, float]:
        """The project's PARAM_TARGETS."""

    def before_step(self) -> None:
        """Put Ansys in a known state before a solve."""

    def solve(
        self,
        study,
        targets,
        folder: Path,
        settings: dict,
        design_variables: dict,
        parameter_targets: dict,
    ):
        """Build the chip with ``design_variables``, run the optimization
        against ``parameter_targets`` with its outputs in ``folder``, and
        return the ``DesignAnalysis``. Raises when a check that has to run
        every iteration fails (mode identity)."""

    # Optional:
    # ``converged(analysis) -> {param: (deviation, within tolerance)}``:
    #   the project's own verdict for targets that set no ``tolerance``
    #   (a frequency's deviation in Hz, any other's as a fraction);
    # ``subjects``: the subsystems, ``{key: (label, group)}`` or
    #   ``{key: (label, group, values)}`` -- ``values`` are the
    #   ``{placeholders}`` a template is filled in with for it;
    # ``campaign_dir``: where campaign files are, if not
    #   ``<root>/campaigns``;
    # ``campaigns(tasks) -> [CampaignSpec]``: campaigns built in code;
    # ``resolve(task: TaskSpec, settings) -> list[str]``: the project's
    #   own refusals (a MiniStudy with too few passes, say);
    # ``modes_checked(settings) -> bool``: False when these settings
    #   switch the per-iteration mode-identity check off;
    # ``reload()``: re-import project modules (the GUI's "Reload");
    # ``code_modules``: extra top-level modules a study file may name;
    # ``design()``: the built qiskit-metal design, to check names.
    # ``path_limit``, ``solver_files``, ``path_limit_note``: what the
    # solver writes into a study's folder and the longest path it takes
    # (Ansys: 259 characters) -- a study past it is warned about.
    # ``bridge_settings(solver)``: ``(size, threshold, offset, label)``
    # of the wire bonds the project's solve really puts in, for
    # ``"hfss"`` or ``"q3d"`` (None: the study's own) -- for the picture
    # of a study, when the solve overrides the study's settings.


class QDOBackend:
    """Drives a :class:`QDOProject` through the engine's interface."""

    name = "qdo"
    words = WORDS

    def __init__(self, project: QDOProject):
        # QDO's logger module, on first import, removes the first
        # StreamHandler on the root logger -- a FileHandler is one. Import
        # it now, before a worker's log file or a study's study.log is
        # attached, or the first solve (importing DesignAnalysis) would
        # silently take that handler away.
        import qdesignoptimizer.logger  # noqa: F401  pylint: disable=unused-import

        self.project = project
        self._templates: dict[str, Template] = {}
        self._targets: list[dict] = []
        self._template_stamp: tuple = ()
        self._discovery_problems: list[str] = []
        self.catalog_problems: list[str] = []
        extra = getattr(project, "code_modules", ())
        if extra:
            from . import qdo_files as qf

            qf.allow_modules(*extra)

    @property
    def path_limit(self) -> int | None:
        return getattr(self.project, "path_limit", None)

    @property
    def solver_files(self) -> tuple:
        return tuple(getattr(self.project, "solver_files", ()) or ())

    @property
    def path_limit_note(self) -> str:
        return getattr(self.project, "path_limit_note", "")

    @property
    def out_dir(self) -> Path:
        return Path(self.project.root) / "out"

    # -- catalog ---------------------------------------------------------

    @property
    def ministudy_dir(self) -> Path:
        """Ministudy templates: ``<design>/ministudies/`` by default."""
        return Path(
            getattr(self.project, "ministudy_dir", None)
            or Path(self.project.root) / MINISTUDY_DIR
        )

    @property
    def opt_targets_dir(self) -> Path:
        """Opt-target files: ``<design>/opt_targets/`` by default."""
        return Path(
            getattr(self.project, "opt_targets_dir", None)
            or Path(self.project.root) / TARGETS_DIR
        )

    @property
    def campaign_dir(self) -> Path:
        """Campaign files: ``<design>/campaigns/`` by default."""
        return Path(
            getattr(self.project, "campaign_dir", None)
            or Path(self.project.root) / CAMPAIGN_DIR
        )

    def subjects(self) -> dict[str, dict]:
        """The project's subsystems: ``{key: {label, group, values}}``
        from its ``subjects`` (a label, ``(label, group)`` or ``(label,
        group, values)`` each)."""
        out = {}
        for key, value in (getattr(self.project, "subjects", None) or {}).items():
            if isinstance(value, dict):
                item = dict(value)
            elif isinstance(value, (tuple, list)):
                item = dict(zip(("label", "group", "values"), value))
            else:
                item = {"label": value}
            out[str(key)] = {
                "label": item.get("label") or str(key),
                "group": item.get("group") or "",
                "values": dict(item.get("values") or {}),
            }
        return out

    def project_targets(self) -> list[dict]:
        """The project's opt-target entries (named, unique), as its
        files hold them."""
        self.templates()
        return list(self._targets)

    def templates(self) -> dict[str, Template]:
        """Every queueable study by key, from the project's templates.
        Re-read whenever a template file changed."""
        files = template_files(self.ministudy_dir, self.opt_targets_dir)
        subjects = {k: v["values"] for k, v in self.subjects().items()}
        stamp = (
            tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files),
            repr(subjects),
        )
        if stamp != self._template_stamp:
            found, problems = discover_templates(
                self.ministudy_dir, subjects, self.template_values()
            )
            self._targets, more = load_targets(self.opt_targets_dir)
            problems.extend(more)
            templates: dict[str, Template] = {}
            for template in found:
                if template.key in templates:
                    problems.append(
                        f"{template.source}: {template.key} is also "
                        f"defined at {templates[template.key].source}"
                    )
                    continue
                templates[template.key] = template
            self._templates = templates
            self._discovery_problems = problems
            self._template_stamp = stamp
        return self._templates

    def catalog(self) -> Catalog:
        templates = self.templates()
        known_subjects = self.subjects()

        def subject_label(key):
            return known_subjects.get(key, {}).get("label") or key

        tasks = []
        for gen in templates.values():
            name = gen.name.replace("_", " ")
            tasks.append(
                TaskSpec(
                    key=gen.key,
                    subject=gen.subject,
                    label=(
                        f"{subject_label(gen.subject)} {name}" if gen.subject else name
                    ),
                    family=gen.family,
                    summary=gen.call_text(),
                )
            )
        subjects = []
        for key in dict.fromkeys(t.subject for t in tasks):
            group = known_subjects.get(key, {}).get("group", "")
            subjects.append(SubjectSpec(key, subject_label(key), group))
        hook = getattr(self.project, "campaigns", None)
        campaigns = list(hook(tasks)) if hook is not None else []
        problems = list(self._discovery_problems)
        specs, more = load_campaign_dir(self.campaign_dir, self.words)
        known = {c.key for c in campaigns}
        campaigns.extend(s for s in specs if s.key not in known)
        problems.extend(more)
        self.catalog_problems = problems
        return Catalog(tasks=tasks, campaigns=campaigns, subjects=subjects)

    def files(self) -> list[FileSpec]:
        return [
            FileSpec(
                MINISTUDY,
                "MiniStudy",
                about="What is simulated: components, ports, pins, "
                "modes, junctions, mesh, capacitance studies.",
            ),
            FileSpec(
                TARGETS,
                "Opt targets",
                about="Which design variable chases which parameter.",
            ),
            FileSpec(
                PARAMS,
                "Parameter targets",
                optional=True,
                about="Targets for this study only; the rest come "
                "from the project when it starts.",
            ),
            FileSpec(
                VARIABLES,
                "Design variables",
                optional=True,
                about="Values for this study only; the rest come "
                "from the project when it starts.",
            ),
        ]

    def settings_keys(self) -> set[str]:
        return set(self.project.settings_keys())

    def check_names(self) -> set[str]:
        return {"converged", "mode_identity"}

    def reload(self) -> None:
        """Read every template again and call the project's own
        ``reload`` hook, for the GUI's "Reload catalog"."""
        self._template_stamp = ()
        hook = getattr(self.project, "reload", None)
        if hook is not None:
            hook()

    def default_settings(self, task: str) -> dict:
        spec = self.catalog().task(task)
        # an unknown task reaches the project's hook as None, as before
        return dict(self.project.default_settings(spec))  # type: ignore[arg-type]

    # -- generating ------------------------------------------------------

    def generate(self, task: str) -> tuple[dict[str, str], str]:
        template = self.templates().get(task)
        if template is None:
            raise KeyError(f"no template {task!r}")
        return self._generate_from_template(task, template)

    def ministudy(self, task: str):
        """The ``MiniStudy`` a template makes now, filled in exactly as
        it would be when queued: for a project's own checks and scripts
        (mesh coverage, mode pictures) that need the study object."""
        from . import qdo_files as qf

        files, _ = self.generate(task)
        return qf.load_object(yaml.safe_load(files[MINISTUDY]))

    def _generate_from_template(
        self, task: str, template: Template
    ) -> tuple[dict[str, str], str]:
        """Fill a ministudy template for one subsystem and give it the
        project's opt targets that fit; the MiniStudy is rebuilt (and
        written as rebuilt) and every target built, so a template that
        does not make a study is refused when it is queued."""
        from . import qdo_files as qf

        ministudy, candidates = render(
            template,
            self.project_targets(),
            self.project.parameter_targets(),
            self.store_values(),
            self.template_values(),
        )
        built = qf.load_object(ministudy)
        study = qf.to_file_data(built)
        entries = study_targets(
            built,
            candidates,
            self.project.parameter_targets(),
            self._design_components(),
        )
        by = template.call_text()
        when = _now()
        return {
            MINISTUDY: dump_yaml(
                study, _MINISTUDY_HEADER.format(task=task, when=when, by=by)
            ),
            TARGETS: dump_yaml(
                {"targets": entries},
                _TARGETS_HEADER.format(task=task, when=when),
            ),
        }, by

    def save_template(self, folder: Path, job, name: str) -> list[Path]:
        """Write a study's MiniStudy as ``ministudies/<name>.yaml`` ("Save
        as template"), and those of its opt targets the project does not
        have yet as ``opt_targets/<name>.yaml``.

        The file is ``<subsystem>_<name>.yaml`` when the study has a
        subsystem of the project, so the template stays with it
        (``<subsystem>.<name>``). A target is the project's by its name
        (``_new_targets``). The template is for one subsystem, names as
        they are; to make it a family, put ``{placeholders}`` where the
        names differ.
        """
        folder = Path(folder)
        if job.subject and job.subject in self.subjects():
            name = f"{job.subject}_{name}" if name else name
        ministudy = yaml.safe_load((folder / MINISTUDY).read_text(encoding="utf-8"))
        entries = (
            yaml.safe_load((folder / TARGETS).read_text(encoding="utf-8")) or {}
        ).get("targets") or []
        from .qdo_templates import check_name

        # A pair, or nothing: the ministudy's name is checked first and
        # the targets, which can still be refused, are written first.
        check_name(name)
        if (self.ministudy_dir / f"{name}.yaml").exists():
            raise TemplateError(
                f"ministudies/{name}.yaml already exists; pick another name"
            )
        new = self._new_targets(entries, name)
        written = []
        if new:
            written.append(write_targets(self.opt_targets_dir, name, new))
        written.insert(0, write_ministudy(self.ministudy_dir, name, ministudy))
        return written

    def _new_targets(self, entries: list, name: str) -> list[dict]:
        """A study's opt targets the project does not have yet: by name,
        or, for one without a name, by content. A nameless one is named
        ``<name>`` (``<name>_<n>`` when there are several).

        The project "has" a target as its file holds it and as it is
        filled in for any subsystem: a study's ``R1_freq`` is the
        project's ``{subject}_freq``, not a new target (saved again it
        would tune the same design variable twice).
        """
        project = self._target_forms()
        taken = {e["name"] for e in project}

        def body(entry):
            return {k: v for k, v in entry.items() if k not in ("name", "active")}

        known = [body(e) for e in project]
        new = [
            dict(e)
            for e in entries
            if isinstance(e, dict)
            and e.get("name") not in taken
            and (e.get("name") or body(e) not in known)
        ]
        several = sum(1 for e in new if not e.get("name")) > 1
        out, n = [], 0
        for entry in new:
            if not entry.get("name"):
                n += 1
                given = f"{name}_{n}" if several else name
                entry = {"name": given, **entry}
            out.append(entry)
        return out

    def _target_forms(self) -> list[dict]:
        """Every project opt target as its file holds it, and as filled
        in for each subsystem that sets its placeholders."""
        from .qdo_templates import fill

        base = self.template_values()
        p, v = self.project.parameter_targets(), self.store_values()
        value_sets = [dict(base)] + [
            {**base, **subject["values"], "subject": key}
            for key, subject in self.subjects().items()
        ]
        out = []
        for entry in self.project_targets():
            out.append(entry)
            for values in value_sets:
                try:
                    out.append(fill(entry, values, p, v))
                except TemplateError:
                    continue  # not for this subsystem
        return out

    # -- checking --------------------------------------------------------

    def check_folder(self, folder: Path, job, deep: bool) -> list[str]:
        folder = Path(folder)
        problems: list[str] = []
        texts = {}
        for name in (MINISTUDY, TARGETS):
            try:
                texts[name] = yaml.safe_load(
                    (folder / name).read_text(encoding="utf-8")
                )
            except FileNotFoundError:
                problems.append(f"{name} is missing")
            except yaml.YAMLError as exc:
                problems.append(f"{name} is not valid YAML: {exc}")
        if TARGETS in texts:
            # Plain YAML, so checked on every pass rather than only in the
            # deep check: a study made from scratch starts with none, and
            # launching it must not wait for a design build to say so.
            from . import qdo_files as qf

            entries = _targets_of(texts.get(TARGETS))
            if not isinstance(entries, list) or not qf.active_targets(entries):
                problems.append(
                    f"{TARGETS} lists no targets"
                    + (" that are switched on" if entries else "")
                )
        store = self.store_values()
        variables = self._partial(folder / VARIABLES, problems, json_file=True)
        unknown = sorted(set(variables) - set(store))
        if unknown:
            problems.append(f"{VARIABLES}: the project has no {unknown}")
        problems.extend(_unit_problems(variables, store))
        params = self._partial(folder / PARAMS, problems, json_file=False)
        known_values: dict = {}
        try:
            known_values = dict(self.project.parameter_targets())
            known_params = set(known_values)
        except Exception as exc:  # noqa: BLE001
            known_params = set(params)
            problems.append(f"cannot read the project's targets: {exc}")
        unknown = sorted(set(params) - known_params)
        if unknown:
            problems.append(f"{PARAMS}: the project has no target {unknown}")
        for key, value in params.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"{PARAMS}: {key} is not a number")
        problems.extend(
            _mode_problems(texts.get(MINISTUDY), params, store, known_params, variables)
        )
        refuse = getattr(self.project, "resolve", None)
        if refuse is not None:
            spec = self.catalog().task(job.task) or self._custom_spec(
                job, texts.get(MINISTUDY)
            )
            if spec is not None:
                problems.extend(refuse(spec, dict(job.settings)) or [])
        if problems or not deep:
            return problems
        return problems + self._deep_problems(
            texts, {**store, **variables}, {**known_values, **params}
        )

    def design_names(self) -> dict[str, set[str]] | None:
        """``{component: {pin, ...}}`` of the built design, or None when
        the project has no ``design()`` hook.

        Building takes seconds (no Ansys), so the result is kept until a
        file in the project's ``design_files()`` changes.

        Raises:
            Exception: Whatever building the design raised.
        """
        build = getattr(self.project, "design", None)
        if build is None:
            return None
        files: list[Any] = getattr(self.project, "design_files", lambda: [])()
        stamp = tuple(
            (str(p), Path(p).stat().st_mtime_ns, Path(p).stat().st_size)
            for p in files
            if Path(p).exists()
        )
        cached = getattr(self, "_design_cache", None)
        if cached is not None and cached[0] == stamp:
            return cached[2]
        design = build()
        names = {
            str(name): set(getattr(component, "pins", {}) or {})
            for name, component in design.components.items()
        }
        self._design_cache = (stamp, design, names)
        return names

    def _design_components(self) -> set[str] | None:
        """Every component name of the built design, or None when the
        project has no ``design()`` or it does not build."""
        try:
            names = self.design_names()
        except Exception:  # noqa: BLE001 -- the deep check reports it
            return None
        return set(names) if names is not None else None

    def built_design(self):
        """The built design (kept as ``design_names`` keeps it), or None
        when the project has no ``design()`` hook."""
        if self.design_names() is None:
            return None
        return self._design_cache[1]

    def preview(self, folder: Path) -> list[dict[str, Any]]:
        """Pictures of what a study simulates, one per solver run (the
        HFSS eigenmode model, then each chained capacitance study), each
        of only that run's components with the airbridges its solver
        puts in -- see ``qdo_preview``.

        Raises:
            ValueError: If the project cannot build its design.
        """
        from . import qdo_files as qf
        from . import qdo_preview

        design = self.built_design()
        if design is None:
            raise ValueError(
                "this project has no design() hook, so there is no " "design to draw"
            )
        study = qf.load_object(
            yaml.safe_load((Path(folder) / MINISTUDY).read_text("utf-8"))
        )
        hook = getattr(self.project, "bridge_settings", None)
        out = []
        for picture in qdo_preview.pictures(design, study, bonds=hook):
            note = qdo_preview.unbridged_note(design, picture["components"])
            picture["caption"] = "; ".join(
                n
                for n in (
                    picture["caption"],
                    note,
                    "geometry as the project's design stands now "
                    "(this study's own design variables are not applied)",
                )
                if n
            )
            out.append(picture)
        return out

    def _name_problems(self, study) -> list[str]:
        """Components and pins a study names that the design lacks.

        Without this a misspelt component reaches Ansys, which fails on
        it -- or, for a pin, silently renders a different circuit.
        """
        try:
            names = self.design_names()
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            return [
                f"cannot build the design to check names: "
                f"{type(exc).__name__}: {exc}"
            ]
        if names is None:
            return []
        problems: list[str] = []

        def component(name, where):
            if name not in names:
                near = difflib.get_close_matches(str(name), names, n=2)
                hint = f" (did you mean {', '.join(near)}?)" if near else ""
                problems.append(
                    f"{MINISTUDY}: {where}: no component {name!r} in the "
                    f"design{hint}"
                )
                return False
            return True

        def pins(entries, where):
            for entry in entries or []:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                comp, pin = entry[0], entry[1]
                if component(comp, where) and pin not in names[comp]:
                    near = difflib.get_close_matches(str(pin), names[comp], n=2)
                    hint = f" (did you mean {', '.join(near)}?)" if near else ""
                    problems.append(
                        f"{MINISTUDY}: {where}: {comp} has no pin " f"{pin!r}{hint}"
                    )

        for name in getattr(study, "qiskit_component_names", []) or []:
            component(name, "qiskit_component_names")
        pins(getattr(study, "port_list", None), "port_list")
        pins(getattr(study, "open_pins", None), "open_pins")
        pins(getattr(study, "grounded_pins", None), "grounded_pins")
        for n, cap in enumerate(
            getattr(study, "capacitance_matrix_studies", None) or [], 1
        ):
            where = f"capacitance study {n}"
            for name in getattr(cap, "qiskit_component_names", []) or []:
                component(name, where)
            pins(getattr(cap, "open_pins", None), where)
        return problems

    def _deep_problems(
        self, texts: dict, variables: dict, params: dict | None = None
    ) -> list[str]:
        from . import qdo_files as qf

        problems = []
        study = None
        try:
            study = qf.load_object(texts[MINISTUDY])
            if not hasattr(study, "qiskit_component_names"):
                problems.append(f"{MINISTUDY} does not build a MiniStudy")
                study = None
            else:
                problems.extend(self._name_problems(study))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{MINISTUDY}: {type(exc).__name__}: {exc}")
        produced = study_parameters(study) if study is not None else None
        switched_on: list = []
        entries = _targets_of(texts.get(TARGETS))
        if not isinstance(entries, list) or not qf.active_targets(entries):
            return problems + [
                f"{TARGETS} lists no targets"
                + (" that are switched on" if entries else "")
            ]
        for n, entry in enumerate(entries, 1):
            if not qf.is_active(entry):
                continue  # kept in the file, not used
            try:
                built = qf.entry_targets(entry)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{TARGETS} target {n}: {exc}")
                continue
            switched_on.extend((n, t) for t in built)
            for target in built:
                par = _target_parameter(target)
                if params is not None and par and par not in params:
                    problems.append(
                        f"{TARGETS} target {n}: {par!r} has no target value "
                        f"(parameter_targets.py or {PARAMS})"
                    )
            if produced is not None:
                for target in built:
                    par = _target_parameter(target)
                    # A capacitance entry's conductors are only known to
                    # Ansys: not refused on a guess (QDO warns at solve).
                    if not _is_capacitance_target(target) and par not in produced:
                        problems.append(
                            f"{TARGETS} target {n}: this study does not "
                            f"measure {par!r}; switch the target off or "
                            f"change the MiniStudy"
                        )
            for target in built:
                var = getattr(target, "design_var", None)
                if var and var not in variables:
                    problems.append(
                        f"{TARGETS} target {n}: no design variable " f"{var!r}"
                    )
                prop = getattr(target, "prop_to", None)
                for key in getattr(getattr(prop, "expression", None), "keys", {}).get(
                    "v", ()
                ):
                    if key not in variables:
                        problems.append(
                            f"{TARGETS} target {n}: prop_to reads "
                            f"v[{key!r}], which does not exist"
                        )
        problems.extend(f"{TARGETS}: {p}" for p in active_target_clashes(switched_on))
        return problems

    @staticmethod
    def _partial(path: Path, problems: list, json_file: bool) -> dict:
        if not path.exists():
            return {}
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text) if json_file else yaml.safe_load(text)
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            problems.append(f"{path.name} cannot be read: {exc}")
            return {}
        if data is None:
            return {}
        if not isinstance(data, dict):
            problems.append(f"{path.name} must hold a mapping")
            return {}
        if not json_file:
            # PyYAML follows YAML 1.1, where 4.9e9 (no sign after the e)
            # is a string; a target typed that way must still be 4.9 GHz.
            data = numbers(data)
        return data

    # -- store -----------------------------------------------------------

    def store_values(self) -> dict[str, Any]:
        return read_json(self.project.design_variables_file)

    def commit(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Write the values; return those that changed."""
        return update_json(self.project.design_variables_file, updates)

    def note_commit(self, updates, previous, source) -> None:
        """Record which study wrote each design variable, next to the
        file: ``design_variables.writes.json``, with the simulation that
        produced it and the target it chased (``write_details``)."""
        try:
            folder = Path(source.get("folder", ""))
            if not folder.is_absolute():
                folder = Path(self.project.root) / folder
            details = self.write_details(folder, list(updates), source.get("settings"))
        except Exception:  # noqa: BLE001 -- the study and its writer
            details = {}  # are recorded regardless
        record_writes(
            self.project.design_variables_file,
            updates,
            previous,
            source,
            details,
        )

    def write_details(
        self, folder: Path, names: list[str], settings: dict | None = None
    ) -> dict:
        """What to keep with each written design variable, read from the
        study's folder as it ran.

        Per variable: ``simulation`` -- ``kind`` (eigenmode when an HFSS
        eigenmode solve ran, else capacitance), the solver ``runs`` in
        order, and the adaptive ``passes`` (``MiniStudy.nbr_passes``)
        -- and ``target``: the
        parameter it chased (``parameter``, ``type``, ``modes``), the
        ``target`` value, the ``achieved`` value of the last iteration and
        whether that is ``within_tolerance``.
        """
        from . import qdo_files as qf
        from . import qdo_preview

        folder = Path(folder)
        study = qf.load_object(yaml.safe_load((folder / MINISTUDY).read_text("utf-8")))
        runs = qdo_preview.runs(study)
        simulation = {
            "kind": (
                "eigenmode"
                if any(r["solver"] == "hfss" for r in runs)
                else "capacitance"
            ),
            "runs": [r["title"] for r in runs],
            "passes": getattr(study, "nbr_passes", None),
        }
        entries = qf.active_targets(
            (yaml.safe_load((folder / TARGETS).read_text("utf-8")) or {}).get("targets")
            or []
        )
        params_path = folder / PARAMS
        params = (
            yaml.safe_load(params_path.read_text("utf-8")) or {}
            if params_path.exists()
            else {}
        )
        result_path = folder / "result.json"
        outputs = (
            (json.loads(result_path.read_text("utf-8")).get("outputs") or {})
            if result_path.exists()
            else {}
        )
        off = set(outputs.get("off_target") or [])
        unjudged = set(outputs.get("unjudged") or [])
        out = {}
        for name in names:
            item: dict[str, Any] = {"simulation": simulation}
            for target in _targets_by_variable(qf, entries).get(name, []):
                parameter = _target_parameter(target)
                item.setdefault("target", []).append(
                    {
                        "parameter": parameter,
                        "type": getattr(target, "target_param_type", None),
                        "modes": [
                            str(m) for m in getattr(target, "involved_modes", [])
                        ],
                        "target": params.get(parameter),
                        "achieved": outputs.get(parameter),
                        "within_tolerance": (
                            None
                            if "off_target" not in outputs or parameter in unjudged
                            else parameter not in off
                        ),
                    }
                )
            out[name] = item
        return out

    def value_sources(self, name: str) -> dict:
        """Which study last wrote each design variable."""
        if name != VARIABLES:
            return {}
        return sources_text(self.project.design_variables_file)

    # -- running ---------------------------------------------------------

    # -- studies made from scratch, and what can be added to a file ------

    #: What a blank study is.
    STUDY_CLASS = "qdesignoptimizer.design_analysis_types:MiniStudy"
    #: The base class of the studies a MiniStudy chains.
    CAPACITANCE_MODULE = "qdesignoptimizer.sim_capacitance_matrix"
    #: Where QDO's opt-target helpers live.
    HELPER_MODULE = "qdesignoptimizer.utils.optimization_targets"

    def _custom_spec(self, job, ministudy) -> TaskSpec | None:
        """A made-from-scratch study's family, read off its MiniStudy
        (capacitance only, or an eigenmode solve), so the project's own
        rules (``resolve``) still apply to it."""
        if not is_custom(job.task):
            return None
        args = (ministudy or {}).get("args") or {}
        family = (
            "capacitance" if args.get("run_capacitance_studies_only") else "eigenmode"
        )
        return TaskSpec(job.task, job.subject, job.task, family=family)

    def template_values(self) -> dict:
        """Named values the project supplies to every template (its
        optional ``template_values()`` hook), {} when it has none."""
        hook = getattr(self.project, "template_values", None)
        return dict(hook() or {}) if hook is not None else {}

    def blank_files(self) -> dict[str, str]:
        """A study from scratch: a MiniStudy with every argument present
        (the required ones blank) and no opt targets yet."""
        from . import qdo_files as qf

        return {
            MINISTUDY: dump_yaml(
                qf.blank_object(self.STUDY_CLASS),
                "MiniStudy made from scratch: fill in the components, "
                "ports, pins and modes,\nthen add capacitance studies "
                "if it chains any.",
            ),
            TARGETS: dump_yaml(
                {"targets": []},
                "Opt targets: add them from QDO's helpers, as explicit "
                "targets, or from the\nproject's own (opt_targets/).",
            ),
        }

    def item_choices(self, name: str) -> dict[tuple, list[dict]]:
        """What can be added where in a study file:
        ``{list path: [{"value", "label"}]}``."""
        import importlib
        import inspect

        if name == MINISTUDY:
            module = importlib.import_module(self.CAPACITANCE_MODULE)
            base = getattr(module, "CapacitanceMatrixStudy", None)
            classes = [
                c
                for _, c in inspect.getmembers(module, inspect.isclass)
                if c.__module__ == module.__name__
                and (base is None or issubclass(c, base))
                and not inspect.isabstract(c)  # cannot be built
            ]
            return {
                ("args", "capacitance_matrix_studies"): [
                    {
                        "value": f"class:{c.__module__}:{c.__qualname__}",
                        "label": c.__name__,
                    }
                    for c in classes
                ]
            }
        if name == TARGETS:
            module = importlib.import_module(self.HELPER_MODULE)
            helpers = [
                {
                    "value": f"helper:{module.__name__}:{n}",
                    "label": n,
                }
                for n, f in inspect.getmembers(module, inspect.isfunction)
                if n.startswith("get_opt_target")
                and f.__module__ == module.__name__
                and _returns_one_target(f)
            ]
            return {
                ("targets",): helpers
                + [
                    {
                        "value": "explicit",
                        "label": "explicit OptTarget (prop_to as an " "expression)",
                    }
                ]
            }
        return {}

    def make_item(self, name: str, choice: str) -> dict:
        """A new, blank list item for ``item_choices``' ``choice``."""
        from . import qdo_files as qf

        if choice == "explicit":
            return qf.blank_target()
        kind, _, ref = choice.partition(":")
        if kind == "class":
            return qf.blank_object(ref)
        if kind == "helper":
            return qf.blank_helper(ref)
        raise ValueError(f"cannot add {choice!r}")

    def target_templates(self) -> list[str]:
        """The names of the project's opt targets (for "Add targets")."""
        return [e["name"] for e in self.project_targets()]

    def template_targets(self, name: str) -> list[dict]:
        """One of the project's opt targets, by name, ``=`` expressions
        filled from the project. One written for a family of subsystems
        (``{n}`` placeholders) is refused here: it needs the values a
        subsystem gives it.

        Raises:
            TemplateError: For an unknown name or a placeholder.
        """
        from .qdo_templates import fill

        entry = next((e for e in self.project_targets() if e["name"] == name), None)
        if entry is None:
            raise TemplateError(f"the project has no opt target {name!r}")
        return [
            fill(
                entry,
                self.template_values(),
                self.project.parameter_targets(),
                self.store_values(),
                name,
            )
        ]

    def save_targets_template(self, folder: Path, name: str) -> Path:
        """Add a study's opt targets to the project, as
        ``opt_targets/<name>.yaml``: those it does not have by name yet
        (switched-off ones included, as they are; one without a name
        gets ``<name>_<n>``).

        Raises:
            TemplateError: For a bad name, a file that exists, or when
                the project already has every one of them.
        """
        entries = (
            yaml.safe_load((Path(folder) / TARGETS).read_text("utf-8")) or {}
        ).get("targets") or []
        new = self._new_targets(entries, name)
        if not new:
            raise TemplateError(
                "the project already has every opt target of this study " "(by name)"
            )
        return write_targets(self.opt_targets_dir, name, new)

    def project_values(self, name: str):
        """The project's values a study's optional file overrides: the
        design variables or the parameter targets, as they stand."""
        if name == VARIABLES:
            return self.store_values()
        if name == PARAMS:
            return dict(self.project.parameter_targets())
        return None

    def resolve_inputs(self, folder: Path, job) -> None:
        folder = Path(folder)
        problems: list[str] = []
        variables = self._partial(folder / VARIABLES, problems, True)
        params = self._partial(folder / PARAMS, problems, False)
        if problems:
            raise ValueError("; ".join(problems))
        full_vars = {**self.store_values(), **variables}
        (folder / VARIABLES).write_text(
            json.dumps(full_vars, indent=4) + "\n", encoding="utf-8"
        )
        full_params = {**self.project.parameter_targets(), **params}
        (folder / PARAMS).write_text(
            dump_yaml(
                full_params,
                "Parameter targets this study ran "
                "against (the project's, plus what the queued "
                f"folder set: {sorted(params) or 'nothing'}).",
            ),
            encoding="utf-8",
        )

    def before_step(self, job) -> None:
        self.project.before_step()

    def run(self, folder: Path, job) -> StepResult:
        from . import qdo_files as qf

        folder = Path(folder)
        study = qf.load_object(yaml.safe_load((folder / MINISTUDY).read_text("utf-8")))
        entries = yaml.safe_load((folder / TARGETS).read_text("utf-8"))["targets"]
        targets = qf.build_targets(entries)
        variables = json.loads((folder / VARIABLES).read_text("utf-8"))
        params = yaml.safe_load((folder / PARAMS).read_text("utf-8")) or {}
        with qdo_record.study_log(folder):
            kept = self._keep_pictures(folder)
            analysis = self.project.solve(
                study, targets, folder, dict(job.settings), variables, params
            )
            kept += self._keep_convergence(folder, analysis, targets, params)
        results = getattr(analysis, "optimization_results", []) or []
        measured = results[-1].get("system_optimized_params") or {} if results else {}
        outputs = {k: v for k, v in measured.items() if v is not None}
        # A target's own tolerance decides; the project's hook, when it
        # has one, speaks for the targets that set none.
        hook = getattr(self.project, "converged", None)
        off = dict(hook(analysis) or {}) if hook is not None else {}
        absolute: set[str] = set()
        off.update(judge_targets(entries, outputs, params, absolute))
        chased = {_target_parameter(t) for t in targets} - {None}
        outputs["unjudged"] = sorted(chased - set(off))
        # Iterations in the DesignAnalysis that finished. After an Ansys
        # drop the project may rebuild the analysis mid-run, and the
        # earlier iterations are not in it -- hence the name.
        outputs["iterations_in_final_analysis"] = len(results)
        # Converged means every switched-on target was judged and hit:
        # one with no tolerance (and no verdict from the project's hook)
        # is not known to be on target.
        outputs["converged"] = (
            bool(off) and not outputs["unjudged"] and all(ok for _, ok in off.values())
        )
        outputs["off_target"] = sorted(k for k, (_, ok) in off.items() if not ok)
        tuned = {t.design_var for t in targets if getattr(t, "design_var", None)}
        updates, files = self._moved_variables(analysis, tuned, variables)
        return StepResult(
            outputs=outputs,
            updates=updates,
            files=files + kept,
            message=_deviation_line(off, absolute),
        )

    def _keep_pictures(self, folder: Path) -> list[str]:
        """The study's pictures (``preview``) into ``pictures/``, before
        the solve; never fails the study."""
        try:
            if self.built_design() is None:
                return []
            return qdo_record.save_pictures(folder, self.preview(folder))
        except Exception:  # noqa: BLE001 -- a picture is not the study
            log.warning("no study pictures for %s", folder.name, exc_info=True)
            return []

    def _keep_convergence(self, folder, analysis, targets, params):
        """``convergence.json`` and ``convergence.png`` from the
        analysis; never fails the study."""
        try:
            return qdo_record.save_convergence(
                folder,
                analysis,
                targets,
                params,
                _target_parameter,
                title=folder.name,
            )
        except Exception:  # noqa: BLE001 -- a plot is not the study
            log.warning("no convergence plot for %s", folder.name, exc_info=True)
            return []

    def _moved_variables(
        self, analysis, tuned: set[str], started: dict
    ) -> tuple[dict, list[str]]:
        """Design variables the optimizer moved during this study.

        Two limits. Only the study's own ``design_var``s: QDO's
        ``_design_variables.json`` can hold more than they (the whole
        ``design.variables`` after a first iteration), so a plain diff
        would sweep in every value merely set for this study.
        And only a change against what the study STARTED from: a value
        you set in the folder that the optimizer left alone is yours for
        this study, not a result to write back.
        """
        save_path = getattr(analysis, "save_path", None)
        if not save_path:
            return {}, []
        path = Path(str(save_path) + "_design_variables.json")
        if not path.exists():
            return {}, [str(save_path)]
        proposed = json.loads(path.read_text(encoding="utf-8"))
        current = self.store_values()
        moved = {
            k: v
            for k, v in proposed.items()
            if k in tuned and k in current and not _same_value(started.get(k), v)
        }
        return moved, [str(path)]

    def check(self, name: str, job, result: StepResult) -> CheckResult:
        if name == "converged":
            if result.outputs.get("converged"):
                return CheckResult(True)
            off = result.outputs.get("off_target") or []
            unjudged = result.outputs.get("unjudged") or []
            parts = []
            if off:
                parts.append(f"off target: {', '.join(off)}")
            if unjudged:
                parts.append(
                    f"not judged (the target sets no tolerance): "
                    f"{', '.join(unjudged)}"
                )
            return CheckResult(False, "; ".join(parts) or "no target could be judged")
        if name == "mode_identity":
            # Checked inside solve on every iteration; a mismatch raises
            # there and the study has already failed -- unless the
            # project was told not to check, which must not pass quietly.
            asked = getattr(self.project, "modes_checked", None)
            if asked is not None and not asked(dict(job.settings)):
                return CheckResult(
                    False, "mode identity was switched off " "for this study"
                )
            return CheckResult(True, "checked every iteration")
        return CheckResult(False, f"no check named {name!r}")

    def describe(self, folder: Path, job) -> dict[str, Any]:
        from . import qdo_files as qf

        folder = Path(folder)
        study = qf.load_object(yaml.safe_load((folder / MINISTUDY).read_text("utf-8")))
        targets = qf.build_targets(
            yaml.safe_load((folder / TARGETS).read_text("utf-8"))["targets"]
        )
        return {
            "components": list(getattr(study, "qiskit_component_names", [])),
            "modes": list(getattr(study, "modes", []) or []),
            "grounded pins": [
                list(p) for p in getattr(study, "grounded_pins", None) or []
            ],
            "capacitance studies": len(
                getattr(study, "capacitance_matrix_studies", None) or []
            ),
            "targets": [
                {
                    "param": getattr(t, "target_param_type", ""),
                    "modes": list(getattr(t, "involved_modes", []) or []),
                    "design variable": getattr(t, "design_var", ""),
                    "prop_to": repr(getattr(t, "prop_to", None)),
                }
                for t in targets
            ],
            "settings": dict(job.settings),
        }


#: Parameter types a capacitance study gives (QDO's names); every other
#: type needs the eigenmode solve.
_CAPACITANCE_TYPES = ("kappa", "charge_line_limited_t1", "capacitance")


def study_modes(study) -> set[str]:
    """The modes a MiniStudy is about: its ``modes`` and the ``mode`` of
    every capacitance study it chains."""
    modes = {str(m) for m in getattr(study, "modes", None) or []}
    for chained in getattr(study, "capacitance_matrix_studies", None) or []:
        mode = getattr(chained, "mode", None)
        if mode:
            modes.add(str(mode))
    return modes


def study_parameters(study) -> dict[str, str]:
    """``{parameter: kind}``: what a MiniStudy's solves measure, as QDO
    names them, and the kind of simulation each comes from.

    The eigenmode solve gives every mode's ``freq``, its ``kappa`` when
    the MiniStudy has ports, and the ``nonlin`` of every pair when it has
    a non-linear junction (EPR). A chained decay study gives
    ``param(mode, <its decay type>)`` -- and QDO lets it overwrite the
    eigenmode value, so its kind wins. Capacitance-matrix entries are not
    listed: which ones exist depends on the Ansys conductor names (see
    ``measures_capacitance``).
    """
    from qdesignoptimizer.utils.names_parameters import (
        FREQ,
        KAPPA,
        param,
        param_nonlin,
    )

    out: dict[str, str] = {}
    if not getattr(study, "run_capacitance_studies_only", False):
        modes = [str(m) for m in getattr(study, "modes", None) or []]
        # QDO's kappa is f/Q: only ports give a mode a coupling loss to
        # measure; without them it reads ~0.
        ports = bool(getattr(study, "port_list", None))
        for mode in modes:
            out[param(mode, FREQ)] = "eigenmode"
            if ports:
                out[param(mode, KAPPA)] = "eigenmode"
        # Nonlinearities come from EPR, which QDO runs only for a
        # non-linear junction (``design_analysis`` skips
        # ``junction_type: linear``).
        if _nonlinear_junctions(study):
            for a in modes:
                for b in modes:
                    out[param_nonlin(a, b)] = "eigenmode"
    for chained in getattr(study, "capacitance_matrix_studies", None) or []:
        decay_type = getattr(chained, "get_decay_parameter_type", None)
        decaying = getattr(chained, "mode", None)
        if decay_type is not None and decaying:
            out[param(str(decaying), decay_type())] = "capacitance"
    return out


def _nonlinear_junctions(study) -> bool:
    """Whether QDO will run EPR for this MiniStudy: a ``jj_setup`` entry
    that is not ``junction_type: linear``."""
    setup = getattr(study, "jj_setup", None) or {}
    return any(
        not (isinstance(jj, dict) and jj.get("junction_type") == "linear")
        for jj in setup.values()
    )


def measures_capacitance(study, conductors, design_components=None) -> bool:
    """Whether a chained capacitance study of ``study`` has these Ansys
    conductors in its matrix.

    Conductor names are not known before Ansys draws the model, so this
    goes by qiskit-metal's naming: a component ``name_tee_1_`` is drawn
    as ``<part>_name_tee_1_`` (``prime_cpw_name_tee_1_``), and
    ``ground_main_plane`` is always there. Every conductor must belong to
    a component of the same chained study.

    A conductor belongs to the LONGEST component name it ends with, among
    ``design_components`` (every component of the built design) when
    given: ``pad_ext_res_1`` is ``ext_res_1``'s, not ``res_1``'s. Without
    them only the study's own components are known, and a component whose
    name ends another's can still be mistaken for it.
    """

    def owner(conductor: str, names) -> str | None:
        found = [k for k in names if conductor == k or conductor.endswith("_" + k)]
        return max(found, key=len) if found else None

    for chained in getattr(study, "capacitance_matrix_studies", None) or []:
        components = {
            str(c) for c in getattr(chained, "qiskit_component_names", None) or []
        }
        names = components | {str(c) for c in design_components or ()}
        if all(
            str(c) == "ground_main_plane" or owner(str(c), names) in components
            for c in conductors
        ):
            return True
    return False


def _is_capacitance_target(target) -> bool:
    return str(getattr(target, "target_param_type", "")) == "capacitance"


def target_produced(
    study, entry: dict, built: list, produced: dict, design_components=None
) -> bool:
    """Whether ``study`` measures every parameter an entry's targets
    chase, in the kind of simulation the entry is tuned in
    (``target_simulation``)."""
    kind = target_simulation(entry, built)
    for target in built:
        if _is_capacitance_target(target):
            modes = list(getattr(target, "involved_modes", None) or [])
            if kind != "capacitance" or not measures_capacitance(
                study, modes, design_components
            ):
                return False
            continue
        if produced.get(_target_parameter(target) or "") != kind:
            return False
    return True


def active_target_clashes(numbered) -> list[str]:
    """Two switched-on targets that tune one design variable, or chase
    one parameter: QDO refuses the first ("Design variables must be
    unique") and the second can never satisfy both.

    Args:
        numbered: ``(entry number, OptTarget)`` pairs.
    """
    problems = []
    seen_vars: dict[str, int] = {}
    seen_params: dict[str, int] = {}
    for n, target in numbered:
        var = getattr(target, "design_var", None)
        if var:
            if var in seen_vars:
                problems.append(
                    f"switched-on targets {seen_vars[var]} and {n} both tune "
                    f"{var!r}; switch one off"
                )
            seen_vars.setdefault(var, n)
        par = _target_parameter(target)
        if par:
            if par in seen_params:
                problems.append(
                    f"switched-on targets {seen_params[par]} and {n} both "
                    f"chase {par!r}; switch one off"
                )
            seen_params.setdefault(par, n)
    return problems


def target_simulation(entry: dict, built: list) -> str:
    """The kind of simulation an opt target is tuned in: its
    ``simulation:`` field, else by its parameter type."""
    stated = entry.get("simulation")
    if stated:
        return str(stated)
    kinds = {str(getattr(t, "target_param_type", "")) for t in built}
    return "capacitance" if kinds and kinds <= set(_CAPACITANCE_TYPES) else "eigenmode"


def study_targets(
    study,
    candidates: list[dict],
    params: dict | None = None,
    design_components=None,
) -> list[dict]:
    """The opt-target entries a study takes from the project's: every
    one whose modes are all modes of its MiniStudy. One the study's
    simulations cannot produce is taken with ``active: false``.

    Args:
        study: The built MiniStudy.
        candidates: The project's entries, filled in for the study.
        params: The project's parameter targets. A target whose
            parameter has no value there starts switched off: QDO would
            fail on it after the first solve.
        design_components: Every component of the built design, when
            known, for ``measures_capacitance``.

    Raises:
        TemplateError: If an entry does not build an OptTarget.
    """
    from . import qdo_files as qf

    modes, produced = study_modes(study), study_parameters(study)
    out, names = [], set()
    on_vars: set[str] = set()
    on_params: set[str] = set()
    for entry in candidates:
        try:
            built = qf.entry_targets(entry)
        except Exception as exc:
            raise TemplateError(f"opt target {entry.get('name')!r}: {exc}") from exc
        if all(_is_capacitance_target(t) for t in built):
            # Its "modes" are Ansys conductors: taken by a study whose
            # chained capacitance study draws them.
            if not all(
                measures_capacitance(
                    study, getattr(t, "involved_modes", None) or [], design_components
                )
                for t in built
            ):
                continue
        else:
            involved = {
                str(m)
                for target in built
                for m in getattr(target, "involved_modes", None) or []
            }
            if not involved or not involved <= modes:
                continue
        if entry.get("name") in names:
            # Unique as written, the same once filled in ("{r}_freq" and
            # "x_freq"): the study would chase one parameter twice.
            raise TemplateError(
                f"two opt targets are called {entry.get('name')!r} once "
                f"filled in for this study; names are unique in the project"
            )
        names.add(entry.get("name"))
        entry = dict(entry)
        variables = {str(v) for t in built if (v := getattr(t, "design_var", None))}
        parameters = {p for t in built if (p := _target_parameter(t))}
        if qf.is_active(entry) and (
            not target_produced(study, entry, built, produced, design_components)
            or (params is not None and not parameters <= set(params))
            # The first target that tunes a variable (or chases a
            # parameter) has it; a second would make QDO refuse the study.
            or variables & on_vars
            or parameters & on_params
        ):
            entry[qf.ACTIVE] = False
        if qf.is_active(entry):
            on_vars |= variables
            on_params |= parameters
        out.append(entry)
    return out


def judge_targets(
    entries: list, measured: dict, params: dict, absolute: set | None = None
) -> dict[str, tuple[float, bool]]:
    """``{param: (deviation, within tolerance)}`` for every switched-on
    entry that sets a ``tolerance``: a frequency's deviation in Hz, and
    so is any parameter judged by ``{abs: ...}`` (its name goes into
    ``absolute``), any other's as a fraction. A parameter that was not
    measured, or a relative tolerance on a target of 0, is off with a
    NaN deviation."""
    from . import qdo_files as qf

    out = {}
    for entry in qf.active_targets(entries):
        try:
            tolerance = qf.tolerance_of(entry)
            if tolerance is None:
                continue
            targets = qf.entry_targets(entry)
        except Exception:  # noqa: BLE001 -- it did not run either
            continue
        kind, limit = tolerance
        for target in targets:
            parameter = _target_parameter(target)
            if parameter is None:
                continue
            wanted, got = params.get(parameter), measured.get(parameter)
            # numpy floats count: QDO's results come out of pandas.
            numbers = all(
                isinstance(x, Real) and not isinstance(x, bool) for x in (wanted, got)
            )
            if not numbers or (not wanted and kind == "rel"):
                out[parameter] = (float("nan"), False)
                continue
            assert got is not None and wanted is not None  # both are numbers
            difference = float(got) - float(wanted)
            relative = float(got) / float(wanted) - 1.0 if wanted else float("nan")
            ok = abs(difference if kind == "abs" else relative) <= limit
            shown_absolute = kind == "abs" or parameter.endswith("_freq")
            if shown_absolute and absolute is not None:
                absolute.add(parameter)
            out[parameter] = (
                difference if shown_absolute else relative,
                bool(ok),
            )
    return out


def _now() -> str:
    from ..engine.state import now

    return now()


def _deviation_line(
    off: dict[str, tuple[float, bool]], absolute: set | None = None
) -> str:
    """One line of every judged parameter's deviation: a frequency in
    MHz, another parameter judged absolutely as a plain number, the rest
    in percent."""
    parts = []
    for name, (dev, ok) in sorted(off.items()):
        if isinstance(dev, float) and math.isnan(dev):
            text = (
                f"{name} not judged (not measured, or a relative "
                f"tolerance on a target of 0: give {{abs: ...}})"
            )
        elif name.endswith("_freq"):
            text = f"{name} {dev / 1e6:+.1f} MHz"
        elif absolute and name in absolute:
            text = f"{name} {dev:+.4g}"
        else:
            text = f"{name} {dev * 100:+.1f}%"
        parts.append(text + ("" if ok else " (off)"))
    return ", ".join(parts)


def param_targets_template() -> str:
    """The text of an empty ``parameter_targets.yaml``."""
    return dump_yaml({}, _PARAMS_HEADER)


def partial_template(name: str) -> str | None:
    """An empty optional file, for the GUI's "Create" button."""
    if name == PARAMS:
        return param_targets_template()
    if name == VARIABLES:
        return "{}\n"
    return None


def _targets_by_variable(qf, entries) -> dict[str, list]:
    """The built OptTargets of a study's file, by their design variable;
    an entry that does not build is left out (it did not run either)."""
    out: dict[str, list] = {}
    for entry in entries:
        try:
            targets = qf.entry_targets(entry)
        except Exception:  # noqa: BLE001
            continue
        for target in targets:
            var = getattr(target, "design_var", None)
            if var:
                out.setdefault(var, []).append(target)
    return out


def _same_value(a: Any, b: Any) -> bool:
    """``"-5um"`` and ``"-5.0 um"`` are one value: the optimizer writes
    numbers back in its own format, which is no move."""
    from .qdo_templates import split_unit

    if a == b:
        return True
    x, y = split_unit(a), split_unit(b)
    return x is not None and y is not None and x == y


def _unit_problems(variables: dict, store: dict) -> list[str]:
    """A study's design variable set in another form than the project's:
    no unit where the project has one (qiskit-metal reads a bare number
    as mm), or another unit (QDO refuses to compare them once it tunes
    the variable, after an iteration has run)."""
    from .qdo_templates import split_unit

    problems = []
    for key in sorted(set(variables) & set(store)):
        mine, theirs = split_unit(variables[key]), split_unit(store[key])
        if theirs is None:
            continue  # not a number the project keeps: nothing to compare
        if mine is None:
            problems.append(
                f"{VARIABLES}: {key} = {variables[key]!r} is not a number "
                f"with a unit (the project has {store[key]!r})"
            )
        elif mine[1] != theirs[1]:
            problems.append(
                f"{VARIABLES}: {key} = {variables[key]!r} has unit "
                f"{mine[1] or 'none'!r}, the project's {store[key]!r} has "
                f"{theirs[1] or 'none'!r}"
            )
    return problems


def _target_parameter(target) -> str | None:
    """The parameter name an OptTarget chases, as QDO names it (as in
    ``design_analysis`` when it merges results)."""
    from qdesignoptimizer.utils.names_parameters import (
        CAPACITANCE,
        NONLIN,
        param,
        param_capacitance,
        param_nonlin,
    )

    kind = getattr(target, "target_param_type", None)
    modes = list(getattr(target, "involved_modes", []) or [])
    try:
        if kind == NONLIN:
            return param_nonlin(*modes)
        if kind == CAPACITANCE:
            return param_capacitance(*modes)
        if kind is None:  # param() refuses it; the except below said None
            return None
        return param(modes[0], kind)
    except Exception:  # noqa: BLE001
        return None


def _targets_of(data) -> list | None:
    """The ``targets:`` list of an opt_targets file's data (None when the
    file is not a mapping that holds one: then it "lists no targets")."""
    if isinstance(data, dict) and isinstance(data.get("targets"), list):
        return data["targets"]
    return None


def _returns_one_target(fn) -> bool:
    """Whether a helper is annotated to return one OptTarget (not a list
    of them: one Add makes one target)."""
    import inspect

    annotation = inspect.signature(fn).return_annotation
    text = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", str(annotation))
    )
    return str(text).split(".")[-1] == "OptTarget"


def _mode_problems(ministudy, params, store, known_params, variables):
    """Modes a study names that no parameter target names.

    The project's modes are read exactly off its parameter targets, as
    QDO names them (``param``, ``param_nonlin``,
    ``param_participation_ratio``: ``resonator_ATS1_freq``,
    ``qubit_a_to_qubit_b_nonlin``, ...), and a mode of an eigenmode
    solve must have a ``<mode>_freq`` target. A mode that is not among them is
    almost always a typo or a name cut short (``resonator_ATS`` for
    ``resonator_ATS1``), which nothing else would catch before the solve.
    Checked for the MiniStudy's ``modes`` and every chained study's
    ``mode``. (``store`` and ``variables`` are not used: a design
    variable's name does not say where its mode ends.)
    """
    del store, variables
    if not isinstance(ministudy, dict):
        return []
    args = ministudy.get("args") or {}
    named = [("modes", m) for m in args.get("modes") or [] if m]
    for n, chained in enumerate(args.get("capacitance_matrix_studies") or [], 1):
        if isinstance(chained, dict) and chained.get("active") is not False:
            mode = (chained.get("args") or {}).get("mode")
            if mode:
                named.append((f"capacitance study {n} mode", mode))
    every = set(known_params) | set(params)
    known = _param_modes(every)
    eigenmode = not args.get("run_capacitance_studies_only")
    problems = []
    for where, mode in named:
        mode = str(mode)
        if mode not in known:
            near = difflib.get_close_matches(mode, sorted(known), n=2)
            hint = f" (did you mean {', '.join(near)}?)" if near else ""
            problems.append(
                f"{MINISTUDY}: {where}: {mode!r} is the mode of no "
                f"parameter target of the project{hint}"
            )
        elif eigenmode and where == "modes" and f"{mode}_freq" not in every:
            # QDO sorts the solved modes by their target frequency, so an
            # eigenmode solve fails for a mode without one -- after it ran.
            problems.append(
                f"{MINISTUDY}: modes: {mode!r} has no target frequency "
                f"{mode + '_freq'!r}, which the eigenmode solve needs"
            )
    return problems


#: Parameter-name endings QDO puts after a mode (``param``).
_MODE_SUFFIXES = ("_freq", "_kappa", "_charge_line_limited_t1")


def _param_modes(names) -> set:
    """The modes in QDO parameter names."""
    out = set()
    for name in map(str, names):
        if name.endswith("_nonlin") and "_to_" in name:
            out.update(name[: -len("_nonlin")].split("_to_", 1))
        elif name.endswith("_participation_ratio") and "_in_" in name:
            out.add(name.split("_in_", 1)[0])
        else:
            for suffix in _MODE_SUFFIXES:
                if name.endswith(suffix):
                    out.add(name[: -len(suffix)])
                    break
    return out
