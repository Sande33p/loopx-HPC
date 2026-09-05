"""PBS/Slurm fixture tests. These never invoke scheduler or SSH executables."""

import json
import subprocess
import unittest

from loopx_hpc.schedulers import (
    parse_status,
    preview_plan,
    render_script,
    validate_resources,
)


class SchedulerRenderTests(unittest.TestCase):
    def setUp(self):
        self.resources = {"job_name": "study-01", "nodes": 2, "walltime": "01:30:00"}

    def test_login_shell_requires_explicit_trusted_startup_request(self):
        self.assertTrue(
            render_script("pbs", ["program"], self.resources, {}).startswith(
                "#!/bin/bash\n"
            )
        )
        for scheduler in ("pbs", "slurm"):
            script = render_script(
                scheduler, ["program"], self.resources, {"login_shell": True}
            )
            self.assertTrue(script.startswith("#!/bin/bash -l\n"))
            self.assertIn("set -euo pipefail", script)
        with self.assertRaises(ValueError):
            render_script("pbs", ["program"], self.resources, {"login_shell": "true"})

    def test_slurm_uses_literal_directives_and_explicit_launcher(self):
        script = render_script(
            "slurm",
            ["srun", "python", "train.py"],
            {
                **self.resources,
                "account": "example-project",
                "queue": "example-partition",
                "ntasks_per_node": 4,
                "cpus_per_task": 2,
                "gpus_per_node": 4,
            },
        )
        for directive in (
            "#SBATCH --nodes=2",
            "#SBATCH --time=01:30:00",
            "#SBATCH --partition=example-partition",
            "#SBATCH --ntasks-per-node=4",
            "#SBATCH --cpus-per-task=2",
            "#SBATCH --gpus-per-node=4",
        ):
            self.assertIn(directive, script)
        self.assertLess(
            script.index("#SBATCH --gpus-per-node"), script.index("set -euo pipefail")
        )
        self.assertTrue(script.endswith("exec -- srun python train.py\n"))

    def test_pbs_uses_no_implicit_site_settings(self):
        minimal = render_script("pbs", ["python", "train.py"], self.resources)
        self.assertIn("#PBS -l select=2\n", minimal)
        for setting in ("system=", "#PBS -A", "#PBS -q", "filesystems=", "mpiexec"):
            self.assertNotIn(setting, minimal)
        explicit = render_script(
            "pbs",
            ["mpiexec", "-n", "8", "program"],
            {
                **self.resources,
                "system": "example-system",
                "account": "example-project",
                "queue": "example-queue",
                "filesystems": ["examplefs", "sharedfs"],
                "place": "scatter",
            },
        )
        self.assertIn("#PBS -l select=2:system=example-system\n", explicit)
        self.assertIn("#PBS -l filesystems=examplefs:sharedfs\n", explicit)
        self.assertIn("#PBS -l place=scatter\n", explicit)

    def test_argv_shell_injection_remains_literal(self):
        literal = "$(printf BAD); `printf WORSE` ' end"
        script = render_script("slurm", ["printf", "%s", literal], self.resources)
        result = subprocess.run(
            ["bash", "-s"], input=script, capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout, literal)

    def test_empty_argument_is_preserved(self):
        script = render_script("pbs", ["printf", "<%s>", ""], self.resources)
        result = subprocess.run(
            ["bash", "-s"], input=script, capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout, "<>")

    def test_rejects_invalid_or_injected_resource_requests(self):
        changes = [
            {"nodes": 0},
            {"nodes": True},
            {"nodes": 1.5},
            {"nodes": 1_000_001},
            {"job_name": "safe\n#SBATCH --nodes=99"},
            {"job_name": "-x"},
            {"queue": "queue --account=other"},
            {"walltime": "00:00:00"},
            {"walltime": "01:99:00"},
            {"walltime": "1-00:00:00"},
            {"walltime": "01:00:00\nexec bad"},
            {"raw_directives": ["--exclusive"]},
            {"gpus_per_node": -1},
            {"system": "other"},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                render_script("slurm", ["true"], {**self.resources, **change})
        for change in (
            {"place": "scatter:group=unconfigured"},
            {"filesystems": "fs"},
            {"filesystems": ["fs", "fs"]},
            {"filesystems": ["fs\nexec bad"]},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_resources("pbs", {**self.resources, **change})

    def test_rejects_invalid_commands_and_unconfigured_environment(self):
        for command in (
            "echo hi",
            [],
            [""],
            ["--help"],
            ["printf", "bad\nline"],
            ["echo", None],
        ):
            with self.subTest(command=command), self.assertRaises(ValueError):
                render_script("slurm", command, self.resources)
        with self.assertRaisesRegex(ValueError, "unconfigured"):
            render_script(
                "pbs", ["true"], self.resources, {"name": "aurora", "configured": False}
            )

    def test_preview_never_enables_execution_and_queries_one_allocation(self):
        plan = preview_plan("slurm", "/scratch/example script.sh", "1234")
        self.assertFalse(plan["execution_enabled"])
        self.assertFalse(plan["site_validated"])
        self.assertEqual(
            plan["submit_argv"], ["sbatch", "--parsable", "/scratch/example script.sh"]
        )
        self.assertIn("--allocations", plan["status_argv"])
        self.assertEqual(plan["cancel_argv"], ["scancel", "1234"])
        pbs = preview_plan("pbs", "/scratch/script.sh", "1234.example-server")
        self.assertEqual(
            pbs["status_argv"],
            ["qstat", "-x", "-f", "-F", "json", "1234.example-server"],
        )
        self.assertIsNone(preview_plan("pbs", "/scratch/script.sh")["status_argv"])
        for job_id in ("--all", "1,2", "1;echo x", "123.batch", "1\n2"):
            with self.subTest(job_id=job_id), self.assertRaises(ValueError):
                preview_plan("slurm", "/scratch/script.sh", job_id)
        with self.assertRaises(ValueError):
            preview_plan("pbs", "--evil", "1")


class SchedulerStatusTests(unittest.TestCase):
    def test_slurm_states_and_native_evidence(self):
        cases = {
            "PENDING": "queued",
            "RUNNING": "running",
            "COMPLETING": "running",
            "COMPLETED": "succeeded",
            "FAILED": "failed",
            "TIMEOUT": "failed",
            "OUT_OF_MEMORY": "failed",
            "CANCELLED by 123": "cancelled",
            "PREEMPTED": "unknown",
            "COMPLETED+": "unknown",
            "INVENTED": "unknown",
        }
        for native, expected in cases.items():
            with self.subTest(native=native):
                parsed = parse_status("slurm", f"1234|{native}|0:0\n")
                self.assertEqual(parsed["status"], expected)
                self.assertEqual(parsed["native_state"], native)
                self.assertEqual(parsed["job_id"], "1234")
                self.assertFalse(parsed["scientific_result_validated"])
        self.assertEqual(
            parse_status("slurm", "1234|COMPLETED|1:0")["status"], "failed"
        )
        self.assertEqual(
            parse_status("slurm", "1234|COMPLETED|0:9")["status"], "failed"
        )

    def test_pbs_states_and_missing_terminal_exit(self):
        for native, expected in {
            "Q": "queued",
            "H": "queued",
            "R": "running",
            "E": "running",
            "S": "running",
            "F": "succeeded",
            "X": "unknown",
            "M": "unknown",
        }.items():
            with self.subTest(native=native):
                output = json.dumps(
                    {
                        "Jobs": {
                            "1234.example-server": {
                                "job_state": native,
                                "Exit_status": 0,
                            }
                        }
                    }
                )
                parsed = parse_status("pbs", output)
                self.assertEqual(parsed["status"], expected)
                self.assertFalse(parsed["scientific_result_validated"])
        for exit_code, expected in (
            (None, "unknown"),
            (1, "failed"),
            (265, "failed"),
            (271, "failed"),
            (-1, "failed"),
            ("0", "unknown"),
            (False, "unknown"),
        ):
            with self.subTest(exit_code=exit_code):
                job = {"job_state": "F"}
                if exit_code is not None:
                    job["Exit_status"] = exit_code
                self.assertEqual(
                    parse_status("pbs", json.dumps({"Jobs": {"12.server": job}}))[
                        "status"
                    ],
                    expected,
                )

    def test_malformed_and_ambiguous_output_never_implies_success(self):
        slurm = [
            "",
            "COMPLETED",
            "123|COMPLETED|",
            "123|COMPLETED|false",
            "123|COMPLETED|0:0|extra",
            "123.batch|COMPLETED|0:0",
            "123|COMPLETED|0:0\n124|COMPLETED|0:0",
            "JobIDRaw|State|ExitCode\n123|COMPLETED|0:0",
        ]
        pbs = [
            "",
            "not JSON",
            "[]",
            "null",
            "{}",
            '{"Jobs": {}}',
            '{"Jobs": {"1": {}, "2": {}}}',
            '{"Jobs": {"1": null}}',
            '{"Jobs": {"1": {"job_state": true}}}',
            '{"Jobs": {"1": {"job_state": "F"}}}',
            '{"Jobs": {"1": {"job_state": "F", "Exit_status": 2, "Exit_status": 0}}}',
            "[" * 2000 + "]" * 2000,
        ]
        for scheduler, outputs in (("slurm", slurm), ("pbs", pbs)):
            for output in outputs:
                with self.subTest(scheduler=scheduler, output=output):
                    parsed = parse_status(scheduler, output)
                    self.assertEqual(parsed["status"], "unknown")
                    self.assertFalse(parsed["terminal"])
        with self.assertRaises(ValueError):
            parse_status("unsupported", "")


if __name__ == "__main__":
    unittest.main()
