"""What a study simulates, as choices: which of the design's components
each solver run includes, and how each of their loose pins ends.

The design is the project's built chip (``QDOProject.design``). The
runs are the MiniStudy's own (its HFSS eigenmode model) and each chained
capacitance study, read straight from the queued ``ministudy.yaml`` so
the GUI can set them where they are (its form of the file).

A pin of an included component ends in one of three ways:

* **short** -- not listed anywhere: qiskit-metal's renderer runs the
  centre trace into the ground plane (the default);
* **open** -- in ``open_pins``: the renderer cuts an endcap there;
* **port** -- in ``port_list`` with an impedance (HFSS only): a lumped
  port, 50 Ohm unless set otherwise.

A pin joined (in the design's netlist) to another included component is
neither: it is a connection inside the model, and is not listed.

A pin joined to a component the run leaves out is a **cut**: the line
goes on in the chip but not in the model. In the HFSS model a cut must
end grounded or in a port, never open -- an open end where the chip has
a line moves the modes (``open_cuts``; the deep study check refuses
it). An end the chip itself has (a pin joined to nothing) keeps
whatever the study gives it. A capacitance (Q3D) study opens cuts on
purpose: grounded, a conductor would join ground. A study that means an
open cut says so in its file::

    extra:
      deliberate_open_cuts: [[component, pin]]

Imports nothing heavy at module scope.
"""

from __future__ import annotations

from typing import Any

SHORT, OPEN, PORT = "short", "open", "port"
STATES = (SHORT, OPEN, PORT)
DEFAULT_OHM = 50

_CHAINED = "capacitance_matrix_studies"


def pin_key(component: str, pin: str) -> str:
    """One pin as one string (component names hold no ``/``)."""
    return f"{component}/{pin}"


def inventory(design) -> list[dict[str, Any]]:
    """Every component of the design: its ``name``, ``class`` and
    ``pins``, each pin with the component it is ``joined`` to ("" when
    loose) and all its ``partners`` (a net may join more than two).
    Sorted by name."""
    partners = _partners(design)
    out = []
    for name in sorted(design.components, key=str):
        component = design.components[name]
        out.append(
            {
                "name": str(name),
                "class": type(component).__name__,
                "pins": [
                    {
                        "name": str(pin),
                        "joined": (partners.get((str(name), str(pin))) or [""])[0],
                        "partners": partners.get((str(name), str(pin)), []),
                    }
                    for pin in (getattr(component, "pins", None) or {})
                ],
            }
        )
    return out


def _partners(design) -> dict[tuple[str, str], list[str]]:
    """Per joined pin ``(component, pin)``: every other component on its
    net in the design's netlist."""
    names_by_id = {
        getattr(component, "id", name): str(name)
        for name, component in design.components.items()
    }
    partners: dict[tuple[str, str], list[str]] = {}
    table = getattr(design, "net_info", None)
    if table is not None and len(table):
        nets: dict[Any, list[tuple[str, str]]] = {}
        for row in table.itertuples(index=False):
            owner = names_by_id.get(row.component_id)
            if owner is not None:
                nets.setdefault(row.net_id, []).append((owner, str(row.pin_name)))
        for ends in nets.values():
            for owner, pin in ends:
                others = [o for o, _ in ends if o != owner]
                if others:
                    partners[(owner, pin)] = others
    return partners


DELIBERATE = "deliberate_open_cuts"


def pairs(entries: Any) -> set[tuple[str, str]]:
    """The ``(component, pin)`` pairs of a pin list (``open_pins``,
    ``port_list``, ...), anything else in it skipped."""
    return {
        (str(e[0]), str(e[1]))
        for e in entries or []
        if isinstance(e, (list, tuple)) and len(e) >= 2
    }


def open_cuts(
    design,
    components: list[str],
    opened: Any,
    ports: Any = None,
    deliberate: Any = None,
) -> list[tuple[str, str, list[str]]]:
    """The cuts a run leaves open: each pin in ``opened`` (its
    ``open_pins``) whose component is in ``components`` and that the
    chip joins to a component the run leaves out -- ``(component, pin,
    the components left out)``. A pin joined to nothing is an end the
    chip has, and may be open; one also in ``ports`` (its ``port_list``)
    ends in the port; one in ``deliberate`` is open on purpose."""
    included = {str(c) for c in components or []}
    partners = _partners(design)
    skip = pairs(ports) | pairs(deliberate)
    out = []
    for entry in opened or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        key = (str(entry[0]), str(entry[1]))
        if key[0] not in included or key in skip:
            continue
        missing = [o for o in partners.get(key, []) if o not in included]
        if missing:
            out.append((key[0], key[1], missing))
    return out


def _short_class(ref: Any) -> str:
    return str(ref or "").split(":")[-1].split(".")[-1]


def runs(data: dict) -> list[dict[str, Any]]:
    """The runs of a ``ministudy.yaml`` (its parsed data) one can set
    components and pins of.

    Returns:
        Per run: ``path`` (where its ``args`` are in the file),
        ``title``, ``solver`` (``hfss`` or ``q3d``), ``components``,
        ``states`` (``{pin_key: [state, ohm]}`` of every pin it lists),
        ``ports`` (whether it takes ports), ``editable`` (False when a
        list is not a plain list, e.g. an expression) and ``note``.
    """
    args = (data or {}).get("args") or {}
    out = [
        _run(
            args,
            ["args"],
            (
                "HFSS eigenmode"
                if args.get("modes") and not args.get("run_capacitance_studies_only")
                else "MiniStudy (solves no eigenmodes)"
            ),
            "hfss",
        )
    ]
    chained = args.get(_CHAINED) or []
    for n, study in enumerate(chained if isinstance(chained, list) else []):
        if not isinstance(study, dict) or not isinstance(study.get("args"), dict):
            continue
        number = f" {n + 1}" if len(chained) > 1 else ""
        run = _run(
            study["args"],
            ["args", _CHAINED, n, "args"],
            f"Q3D capacitance{number}: {_short_class(study.get('class'))}",
            "q3d",
        )
        if study.get("active") is False:
            run["note"] = "switched off: it does not run"
        out.append(run)
    return out


def _run(args: dict, path: list, title: str, solver: str) -> dict[str, Any]:
    components = args.get("qiskit_component_names")
    opened = args.get("open_pins")
    ports = args.get("port_list")
    takes_ports = solver == "hfss" or "port_list" in args
    editable = all(
        value is None or isinstance(value, list)
        for value in (components, opened, ports)
    )
    states: dict[str, list] = {}
    if editable:
        for entry in opened or []:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                states[pin_key(entry[0], entry[1])] = [OPEN, DEFAULT_OHM]
        for entry in ports or []:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                ohm = entry[2] if len(entry) > 2 else DEFAULT_OHM
                states[pin_key(entry[0], entry[1])] = [PORT, ohm]
    return {
        "path": path,
        "title": title,
        "solver": solver,
        "components": [str(c) for c in components or []] if editable else [],
        "states": states,
        "ports": takes_ports,
        "editable": editable,
        "note": (
            ""
            if editable
            else "its components or pins are not a plain list here (an "
            "expression?): type them in the fields as text"
        ),
    }


def markers(design, components: list[str], states: dict[str, list]) -> list[dict]:
    """The pins to mark on a picture of ``components``: every pin of
    each, where it is, and how it ends (``joined`` when it meets
    another included component)."""
    included = set(components)
    joined = {
        (c["name"], p["name"]): p["joined"]
        for c in inventory(design)
        if c["name"] in included
        for p in c["pins"]
    }
    out = []
    for name in components:
        if name not in design.components:
            continue
        component = design.components[name]
        for pin, info in (getattr(component, "pins", None) or {}).items():
            middle = (info or {}).get("middle") if isinstance(info, dict) else None
            if middle is None:
                continue
            partner = joined.get((name, str(pin)), "")
            state = (states.get(pin_key(name, pin)) or [SHORT])[0]
            out.append(
                {
                    "component": name,
                    "pin": str(pin),
                    "x": float(middle[0]),
                    "y": float(middle[1]),
                    "state": "joined" if partner in included else state,
                }
            )
    return out
