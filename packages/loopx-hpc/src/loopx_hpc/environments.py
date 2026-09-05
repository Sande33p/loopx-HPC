"""Validate declarative job environments without reading or changing the host.

Profiles are recipes, not proof of an installed environment. Persist their digest
alongside the resolved runtime/container versions collected by a future executor.
No free-form shell setup, credential lookup, or automatic site defaults are used.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import PurePosixPath
from typing import Any

_FIELDS = {
    "name",
    "configured",
    "modules",
    "variables",
    "working_directory",
    "purge_modules",
    "login_shell",
}
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+/@:-]{0,127}\Z")
_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# These alter shell parsing/startup or overwrite identity-specific roots. This
# explicit policy is narrower than a claim that arbitrary environment is safe.
_RESERVED_VARIABLES = {
    "BASH_ENV",
    "ENV",
    "SHELLOPTS",
    "BASHOPTS",
    "IFS",
    "PS4",
    "BASH_XTRACEFD",
    "PROMPT_COMMAND",
    "HOME",
    "CODEX_HOME",
}


def _text(value: Any, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not empty):
        raise ValueError(
            f"{field} must be a {'possibly empty ' if empty else ''}string"
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field} must not contain control characters")
    return value


def validate_environment(profile: dict) -> dict:
    """Return a detached, normalized recipe; reject unsupported or unsafe fields.

    Optional fields: name, configured, modules, variables, working_directory,
    purge_modules, login_shell. An explicit login_shell=True requests trusted
    user/site Bash startup files; it does not make those files reproducible or
    safe. ``configured=False`` is valid placeholder data, but cannot be
    rendered. Modules are loaded in order; variable exports are sorted. Absolute
    POSIX working directories are not expanded or checked on this local machine.
    """
    if not isinstance(profile, dict):
        raise ValueError("environment profile must be an object")
    if any(not isinstance(key, str) for key in profile):
        raise ValueError("environment field names must be strings")
    unknown = set(profile) - _FIELDS
    if unknown:
        raise ValueError(f"unsupported environment fields: {sorted(unknown)}")
    name = _text(profile.get("name", "custom"), "name")
    if not _TOKEN.fullmatch(name):
        raise ValueError("name must be a compact profile identifier")
    configured = profile.get("configured", True)
    purge = profile.get("purge_modules", False)
    login = profile.get("login_shell", False)
    if any(type(value) is not bool for value in (configured, purge, login)):
        raise ValueError("configured, purge_modules and login_shell must be booleans")
    modules = profile.get("modules", [])
    if not isinstance(modules, list) or len(modules) > 64:
        raise ValueError("modules must be a list with at most 64 entries")
    checked_modules = []
    for module in modules:
        if not _TOKEN.fullmatch(_text(module, "module")):
            raise ValueError("module must be a literal module identifier")
        if module in checked_modules:
            raise ValueError("modules must not contain duplicate entries")
        checked_modules.append(module)
    variables = profile.get("variables", {})
    if not isinstance(variables, dict) or len(variables) > 128:
        raise ValueError("variables must be an object with at most 128 entries")
    checked_variables = {}
    for key, value in variables.items():
        if not isinstance(key, str) or not _VARIABLE.fullmatch(key):
            raise ValueError("environment variable names must be shell identifiers")
        if key in _RESERVED_VARIABLES:
            raise ValueError(f"reserved environment variable: {key}")
        checked_variables[key] = _text(value, f"variables.{key}", empty=True)
    directory = profile.get("working_directory")
    if directory is not None:
        directory = _text(directory, "working_directory")
        path = PurePosixPath(directory)
        if not path.is_absolute() or ".." in path.parts or directory.startswith("//"):
            raise ValueError(
                "working_directory must be an absolute normalized POSIX path"
            )
        directory = str(path)
    return {
        "name": name,
        "configured": configured,
        "modules": checked_modules,
        "variables": dict(sorted(checked_variables.items())),
        "working_directory": directory,
        "purge_modules": purge,
        "login_shell": login,
    }


def environment_digest(profile: dict) -> str:
    """SHA-256 of the canonical declared recipe, not of remote installed software."""
    encoded = json.dumps(
        validate_environment(profile), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def render_environment_prologue(profile: dict) -> list[str]:
    """Produce Bash lines only. A future executor must attest actual resolution."""
    checked = validate_environment(profile)
    if not checked["configured"]:
        raise ValueError(f"environment {checked['name']} is explicitly unconfigured")
    lines = [f"# environment-recipe-sha256: {environment_digest(checked)}"]
    if checked["modules"] or checked["purge_modules"]:
        lines.append(
            "command -v module >/dev/null 2>&1 || { printf '%s\\n' 'module command unavailable' >&2; exit 1; }"
        )
        # Aurora Lmod can inspect unset shell-detection variables. Relax only
        # nounset while loading modules; retain errexit/pipefail and restore the
        # caller's nounset state before any user variable export or workload.
        lines.extend(
            [
                "case $- in *u*) _LOOPX_HPC_RESTORE_NOUNSET=1 ;; *) _LOOPX_HPC_RESTORE_NOUNSET=0 ;; esac",
                "set +u",
            ]
        )
    if checked["purge_modules"]:
        lines.append("module purge")
    lines.extend(f"module load {shlex.quote(module)}" for module in checked["modules"])
    if checked["modules"] or checked["purge_modules"]:
        lines.extend(
            [
                'if [ "$_LOOPX_HPC_RESTORE_NOUNSET" = 1 ]; then set -u; fi',
                "unset _LOOPX_HPC_RESTORE_NOUNSET",
            ]
        )
    lines.extend(
        f"export {key}={shlex.quote(value)}"
        for key, value in checked["variables"].items()
    )
    if checked["working_directory"] is not None:
        lines.append(f"cd -- {shlex.quote(checked['working_directory'])}")
    return lines


def aurora_environment() -> dict:
    """A placeholder only: no guessed modules, queue, project, or filesystem."""
    return validate_environment({"name": "aurora", "configured": False})
