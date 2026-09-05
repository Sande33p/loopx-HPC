"""Declarative environment boundary tests; no remote environment is inspected."""

import subprocess
import unittest

from loopx_hpc.environments import (
    aurora_environment,
    environment_digest,
    render_environment_prologue,
    validate_environment,
)


class EnvironmentTests(unittest.TestCase):
    def test_normalization_is_detached_and_digest_is_order_independent(self):
        profile = {
            "name": "test",
            "variables": {"B": "2", "A": "1"},
            "modules": ["compiler/1.0"],
        }
        checked = validate_environment(profile)
        checked["modules"].append("mpi/2.0")
        self.assertEqual(profile["modules"], ["compiler/1.0"])
        self.assertEqual(
            environment_digest(profile),
            environment_digest(
                {
                    "modules": ["compiler/1.0"],
                    "variables": {"A": "1", "B": "2"},
                    "name": "test",
                }
            ),
        )
        self.assertNotEqual(environment_digest(profile), environment_digest(checked))

    def test_aurora_is_unconfigured_and_cannot_render(self):
        profile = aurora_environment()
        self.assertEqual(profile["name"], "aurora")
        self.assertFalse(profile["configured"])
        self.assertEqual(profile["modules"], [])
        self.assertEqual(profile["variables"], {})
        with self.assertRaisesRegex(ValueError, "unconfigured"):
            render_environment_prologue(profile)

    def test_recipe_has_explicit_module_order_and_directory(self):
        lines = render_environment_prologue(
            {
                "modules": ["compiler/1.0", "mpi/2.0"],
                "purge_modules": True,
                "working_directory": "/scratch/project with space",
            }
        )
        self.assertLess(
            lines.index("module purge"), lines.index("module load compiler/1.0")
        )
        self.assertLess(
            lines.index("module load compiler/1.0"), lines.index("module load mpi/2.0")
        )
        self.assertEqual(lines[-1], "cd -- '/scratch/project with space'")

    def test_shell_metacharacters_are_literal_export_values(self):
        value = "$(printf INJECTED); `printf BAD` ' literal"
        lines = render_environment_prologue({"variables": {"STUDY_LABEL": value}})
        completed = subprocess.run(
            ["bash", "-c", "\n".join(lines) + '\nprintf %s "$STUDY_LABEL"'],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(completed.stdout, value)

    def test_rejects_shell_setup_and_control_injection(self):
        profiles = [
            {"setup_script": "echo arbitrary"},
            {"configured": "false"},
            {"purge_modules": 1},
            {"modules": ["compiler/1; echo injected"]},
            {"modules": ["--force"]},
            {"modules": ["compiler/1", "compiler/1"]},
            {"modules": "compiler/1"},
            {"variables": {"BAD;echo": "x"}},
            {"variables": {"A": "one\ntwo"}},
            {"variables": {"BASH_ENV": "/tmp/startup.sh"}},
            {"variables": {"HOME": "/tmp/identity"}},
            {"variables": {"CODEX_HOME": "/tmp/identity"}},
            {"working_directory": "~/work"},
            {"working_directory": "/scratch/../other"},
            {"working_directory": "/scratch\nexec bad"},
            {"name": "test\n#PBS -q other"},
            {1: "invalid field name", "other": "x"},
        ]
        for profile in profiles:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                validate_environment(profile)

    def test_empty_environment_is_explicitly_minimal_not_site_resolution(self):
        profile = validate_environment({})
        self.assertEqual(profile["modules"], [])
        self.assertIsNone(profile["working_directory"])
        self.assertEqual(len(render_environment_prologue(profile)), 1)


if __name__ == "__main__":
    unittest.main()
