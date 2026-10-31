"""Command line for parameter declarations. Every command prints JSON or a short line; none needs a screen.

``python -m manylatents.params show NAME [--group G]``
    The declaration as JSON on stdout. Exit 0, or 2 for an unknown name.
``python -m manylatents.params dump DIR``
    One ``<group>.<name>.json`` per name of ``names("latent")`` and ``names("lightning")``.
    Exit 0, or 2 when ``DIR`` cannot be written.

To check a declaration file from any producer, run ``python -m manykinds.params check FILE``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from manylatents.params import check_declaration, domain, names


def _show(name: str, group: str) -> int:
    try:
        doc = domain(name, group=group)
    except (KeyError, ValueError) as exc:
        print(f"show: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(doc, indent=2))
    return 0


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
    dump = sub.add_parser("dump", help="write one <group>.<name>.json per method")
    dump.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "show":
        return _show(args.name, args.group)
    return _dump(args.directory)


if __name__ == "__main__":
    sys.exit(main())
