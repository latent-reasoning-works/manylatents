"""Command line for parameter declarations. Every command prints JSON or a short line; none needs a screen.

``python -m manylatents.params show NAME [--group G]``
    The declaration as JSON on stdout. Exit 0, or 2 for an unknown name.
``python -m manylatents.params check FILE``
    ``{"ok": bool, "problems": [{"path", "message"}]}`` on stdout for a JSON or YAML declaration
    from any producer. Exit 0 when valid, 1 when it has problems, 2 when the file is unreadable.
``python -m manylatents.params dump DIR``
    One ``<group>.<name>.json`` per name of ``names("latent")`` and ``names("lightning")``.
    Exit 0, or 2 when ``DIR`` cannot be written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from manylatents.params import check_declaration, domain, names


def _problem(message: str) -> dict:
    path, sep, rest = message.partition(": ")
    if not sep:
        return {"path": "", "message": message}
    return {"path": path, "message": rest}


def _show(name: str, group: str) -> int:
    try:
        doc = domain(name, group=group)
    except (KeyError, ValueError) as exc:
        print(f"show: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(doc, indent=2))
    return 0


def _check(path: Path) -> int:
    import yaml

    try:
        text = path.read_text()
        doc = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
        problem = {"path": "", "message": f"cannot read {path}: {type(exc).__name__}: {exc}"}
        print(json.dumps({"ok": False, "problems": [problem]}))
        return 2
    problems = [_problem(m) for m in check_declaration(doc)]
    print(json.dumps({"ok": not problems, "problems": problems}))
    return 1 if problems else 0


def _dump(directory: Path) -> int:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        count = 0
        for group in ("latent", "lightning"):
            for name in names(group):
                doc = domain(name, group=group)
                problems = check_declaration(doc)
                if problems:
                    print(f"dump: {group}.{name}: " + "; ".join(problems), file=sys.stderr)
                    return 2
                (directory / f"{group}.{name}.json").write_text(json.dumps(doc, indent=2) + "\n")
                count += 1
    except OSError as exc:
        print(f"dump: cannot write {directory}: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {count} declarations to {directory}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m manylatents.params")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show", help="print one declaration as JSON")
    show.add_argument("name")
    show.add_argument("--group", default="latent", choices=["latent", "lightning"])
    check = sub.add_parser("check", help="check a JSON or YAML declaration file")
    check.add_argument("file", type=Path)
    dump = sub.add_parser("dump", help="write one <group>.<name>.json per method")
    dump.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "show":
        return _show(args.name, args.group)
    if args.command == "check":
        return _check(args.file)
    return _dump(args.directory)


if __name__ == "__main__":
    sys.exit(main())
