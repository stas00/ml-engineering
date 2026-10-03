"""CPU-only tests for search geometry and timing defaults."""
import ast
from dataclasses import dataclass, fields
import math
from pathlib import Path
import unittest


# The executable initializes its accelerator at import time. Load only the pure
# definitions so these tests require neither PyTorch nor a GPU.
source = Path(__file__).with_name("mamf-finder.py")
tree = ast.parse(source.read_text())
names = {"wave_mn_layouts", "long_k_shapes", "Tuning"}
module = ast.Module(body=[node for node in tree.body if getattr(node, "name", None) in names], type_ignores=[])
namespace = {"math": math, "dataclass": dataclass, "fields": fields}
exec(compile(module, str(source), "exec"), namespace)
long_k_shapes = namespace["long_k_shapes"]
Tuning = namespace["Tuning"]


class LongKTests(unittest.TestCase):
    def test_h100_bounds_and_determinism(self):
        args = (132, 20480, (128, 256), 32768, 131072, 80 * 2**30)
        shapes = long_k_shapes(*args)
        self.assertEqual(shapes, long_k_shapes(*args))
        self.assertEqual(len(shapes), len(set(shapes)))
        self.assertEqual({k for _, _, k in shapes}, {32768, 65536, 131072})
        self.assertTrue(all(1024 <= min(m, n) <= max(m, n) <= 20480 for m, n, _ in shapes))
        self.assertTrue(all((n, m, k) in shapes for m, n, k in shapes))

    def test_amd_compute_units_and_non_power_of_two_cap(self):
        shapes = long_k_shapes(304, 20480, (128, 128), 32768, 98304, 192 * 2**30)
        self.assertEqual({k for _, _, k in shapes}, {32768, 65536, 98304})

    def test_memory_bound_and_invalid_inputs(self):
        self.assertEqual(long_k_shapes(132, 20480, (128, 256), 32768, 131072, 2**20), [])
        for minimum, maximum, free in [(0, 131072, 80e9), (65536, 32768, 80e9), (32769, 131072, 80e9), (32768, 131072, float("nan"))]:
            with self.assertRaises(ValueError):
                long_k_shapes(132, 20480, (128, 256), minimum, maximum, free)

    def test_profiles_preserve_old_defaults_and_allow_explicit_overrides(self):
        old = Tuning.from_overrides([])
        self.assertEqual((old.mamf_burst_iters, old.mamf_idle_s, old.mamf_screen_idle_s), (20, 0.25, 0.05))
        long = Tuning.from_overrides([], search="long-k")
        self.assertEqual((long.mamf_burst_iters, long.mamf_idle_s, long.mamf_screen_idle_s), (3, 5.0, 5.0))
        custom = Tuning.from_overrides(["mamf_idle_s=10", "long_k_max=262144"], search="long-k")
        self.assertEqual((custom.mamf_idle_s, custom.long_k_max), (10, 262144))
        for override in ["long_k_min=0", "long_k_max=32769", "mamf_idle_s=nan", "mamf_burst_iters=0"]:
            with self.assertRaises(ValueError):
                Tuning.from_overrides([override], search="long-k")


if __name__ == "__main__":
    unittest.main()
