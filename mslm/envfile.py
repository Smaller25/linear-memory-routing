"""Single source of truth for run settings: ``configs/*.env`` files.

The shell scripts (``scripts/lib/config.sh``) and the Python side (this module,
used by the control-tower notebook and the tests) resolve a config the same
way, and ``tests/test_config_parity.py`` checks that they agree:

1. ``configs/base.env`` declares EVERY variable (with a one-line comment).
2. The chosen ``CONFIG`` file is applied on top of it (it may only reassign
   variables that ``base.env`` declares).
3. A variable already present in the caller's environment wins over both
   files ("override"), and later lines that reference it see the override.

Env files are restricted to plain ``KEY=value  # comment`` lines so both
parsers stay trivially equivalent: values are unquoted words, ``"double"``
quoted (``$VAR``/``${VAR}``/``${VAR:-default}`` expanded) or ``'single'``
quoted (literal).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_ENV = REPO_ROOT / "configs" / "base.env"

_ASSIGN = re.compile(r"^(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.*)$")
_SECTION = re.compile(r"^#\s*===\s*\[(\w+)\]")
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")


@dataclass
class EnvLine:
    key: str
    raw_value: str
    comment: str
    section: str
    lineno: int


@dataclass
class EnvFile:
    path: Path
    lines: list[EnvLine] = field(default_factory=list)

    @property
    def keys(self) -> list[str]:
        return [ln.key for ln in self.lines]


def _split_value(rest: str) -> tuple[str, str]:
    """Split ``value  # comment`` into (raw value incl. quotes, comment)."""
    rest = rest.rstrip()
    if rest[:1] in ("'", '"'):
        q = rest[0]
        end = rest.find(q, 1)
        if end < 0:
            raise ValueError(f"unterminated quote in: {rest!r}")
        raw, tail = rest[: end + 1], rest[end + 1 :]
    else:
        m = re.match(r"(\S*)(.*)$", rest)
        raw, tail = m.group(1), m.group(2)
    tail = tail.strip()
    if tail and not tail.startswith("#"):
        raise ValueError(f"unexpected text after value (quote values with spaces): {rest!r}")
    return raw, tail.lstrip("#").strip()


def parse_env_file(path: str | os.PathLike) -> EnvFile:
    path = Path(path)
    out = EnvFile(path)
    section = ""
    for i, line in enumerate(path.read_text().splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        m = _SECTION.match(s)
        if m:
            section = m.group(1)
            continue
        if s.startswith("#"):
            continue
        m = _ASSIGN.match(s)
        if not m:
            raise ValueError(f"{path}:{i}: only KEY=value lines are allowed, got {line!r}")
        raw, comment = _split_value(m.group(2))
        out.lines.append(EnvLine(m.group(1), raw, comment, section, i))
    return out


def _expand(raw: str, scope: dict[str, str]) -> str:
    if raw.startswith("'"):
        return raw[1:-1]
    if raw.startswith('"'):
        raw = raw[1:-1]

    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(3)
        val = scope.get(name, "")
        if m.group(1) and m.group(2) is not None and val == "":
            return _expand(m.group(2), scope)
        return val

    return _VAR.sub(sub, raw)


def declared_keys() -> list[str]:
    return parse_env_file(BASE_ENV).keys


def resolve(config: str | os.PathLike | None = None,
            environ: dict[str, str] | None = None) -> dict[str, str]:
    """Resolve base.env + config (+ overrides from ``environ``) like the shell does.

    Returns only the declared keys, in declaration order.
    """
    environ = dict(os.environ if environ is None else environ)
    base = parse_env_file(BASE_ENV)
    keys = base.keys
    overrides = {k: environ[k] for k in keys if k in environ}
    chosen = {ln.key: ln for ln in base.lines}
    if config is not None and Path(config).resolve() != BASE_ENV.resolve():
        cfg = parse_env_file(config)
        for ln in cfg.lines:
            if ln.key not in chosen:
                raise KeyError(f"{cfg.path}:{ln.lineno}: {ln.key} is not declared in configs/base.env")
            chosen[ln.key] = ln
    # Evaluate in base.env order: each key takes its value from the override,
    # else the config line, else the base line.  So a config that changes
    # SCRATCH_ROOT also moves DATA_ROOT, RUNS_ROOT, ... declared after it.
    scope = dict(environ)
    for key in keys:
        if key not in overrides:
            scope[key] = _expand(chosen[key].raw_value, scope)
    return {k: scope.get(k, "") for k in keys}


def overrides_from(environ: dict[str, str] | None = None) -> dict[str, str]:
    environ = dict(os.environ if environ is None else environ)
    return {k: environ[k] for k in declared_keys() if k in environ}


def set_value(path: str | os.PathLike, key: str, value: str) -> None:
    """Rewrite (or append) ``key`` in an env file, keeping its comment.

    Used by the notebook so that edits land in the same file the shell reads.
    """
    path = Path(path)
    if key not in declared_keys():
        raise KeyError(f"{key} is not declared in configs/base.env")
    if re.search(r"\s", value) or value == "":
        value = f'"{value}"'
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        m = _ASSIGN.match(line.strip())
        if m and m.group(1) == key:
            _, comment = _split_value(m.group(2))
            new = f"{key}={value}"
            lines[i] = f"{new:<40} # {comment}" if comment else new
            break
    else:
        comment = next(ln.comment for ln in parse_env_file(BASE_ENV).lines if ln.key == key)
        lines.append(f"{f'{key}={value}':<40} # {comment}")
    path.write_text("\n".join(lines) + "\n")


def describe() -> list[EnvLine]:
    """Every declared variable with its section and comment (from base.env)."""
    return parse_env_file(BASE_ENV).lines
