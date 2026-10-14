#!/usr/bin/env python3
# Unit tests for the KL gate judging rule (KL-ROBUST-2): pure Python + numpy, no GPU, no exe.
# SPDX-License-Identifier: Apache-2.0
# usage: python tools/klgate/test_klgate_judge.py   (exit 0 = all pass)
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import klgate  # noqa: E402

GROUPS = [8192] * 3 + [32768] * 3
POSS = [0, 1, 2] * 2


def win(end, kl, top1, zero=0):
    return (end, zero, kl, kl * 3, kl * 10, top1)


def noise(kls, top1s):
    return {"n_mean": sum(kls) / len(kls), "windows": [{"kl_mean": k, "top1": t} for k, t in zip(kls, top1s)]}


def by_rule(cases, rule):
    return [c for c in cases if c["rule"] == rule]


class Dense(unittest.TestCase):
    def test_average_first(self):
        # one position over the KL limit (0.006 > 0.005), the length average under it: passes
        res = [win(1, 0.006, 0.99), win(2, 0.004, 0.99), win(3, 0.004, 0.99)] + [win(4, 0.001, 0.99)] * 3
        cs = klgate.judge(res, GROUPS, POSS, "balance", False)
        self.assertEqual(len(cs), 2)  # one case per length, no mean case for dense
        self.assertAlmostEqual(cs[0]["kl_mean"], 0.014 / 3)
        self.assertTrue(all(c["pass"] for c in cs))
        self.assertEqual([p["pos"] for p in cs[0]["positions"]], [0, 1, 2])
        self.assertEqual(cs[0]["positions"][0]["kl_mean"], 0.006)

    def test_average_over_limit_fails(self):
        res = [win(1, 0.006, 0.99), win(2, 0.005, 0.99), win(3, 0.0045, 0.99)] + [win(4, 0.001, 0.99)] * 3
        cs = klgate.judge(res, GROUPS, POSS, "balance", False)
        self.assertFalse(cs[0]["pass"])
        self.assertTrue(cs[1]["pass"])

    def test_top1_average(self):
        # 97.5 / 98.5 / 98.2 -> 98.067 >= 98 passes; 97.5 / 98.2 / 98.2 -> 97.97 fails
        ok = klgate.judge([win(1, 1e-4, .975), win(2, 1e-4, .985), win(3, 1e-4, .982)], GROUPS[:3], POSS[:3], "balance", False)
        bad = klgate.judge([win(1, 1e-4, .975), win(2, 1e-4, .982), win(3, 1e-4, .982)], GROUPS[:3], POSS[:3], "balance", False)
        self.assertTrue(ok[0]["pass"])
        self.assertFalse(bad[0]["pass"])
        self.assertEqual(ok[0]["top1_limit"], 0.98)

    def test_boundary_exact(self):
        # exactly at the limits passes (<= KL, >= top-1), also after float averaging
        cs = klgate.judge([win(1, 0.03, .93)] * 3, GROUPS[:3], POSS[:3], "fast", False)
        self.assertTrue(cs[0]["pass"])
        cs = klgate.judge([win(1, 0.029, .92), win(2, 0.031, .94), win(3, 0.03, .93)], GROUPS[:3], POSS[:3], "fast", False)
        self.assertTrue(cs[0]["pass"])
        cs = klgate.judge([win(1, 0.03001, .93)] * 3, GROUPS[:3], POSS[:3], "fast", False)
        self.assertFalse(cs[0]["pass"])
        cs = klgate.judge([win(1, 0.01, .9299)] * 3, GROUPS[:3], POSS[:3], "fast", False)
        self.assertFalse(cs[0]["pass"])

    def test_zero_rows_fail_the_length(self):
        res = [win(1, 1e-4, .99), win(2, 1e-4, .99, zero=1), win(3, 1e-4, .99)]
        cs = klgate.judge(res, GROUPS[:3], POSS[:3], "balance", False)
        self.assertFalse(cs[0]["pass"])
        self.assertEqual(cs[0]["zero_rows"], 1)

    def test_single_position(self):
        # --positions 1: the average of one window is the window
        cs = klgate.judge([win(8192, 0.0049, .981), win(32768, 0.0051, .99)], [8192, 32768], [0, 0], "balance", False)
        self.assertEqual([c["pass"] for c in cs], [True, False])
        self.assertEqual(cs[0]["kl_mean"], 0.0049)


class MoE(unittest.TestCase):
    N_KL = [0.001, 0.002, 0.003, 0.002, 0.002, 0.002]  # N = 0.002
    N_T1 = [0.99, 0.96, 0.97, 0.98, 0.98, 0.98]        # all-window mean 0.97667; 8192's own mean 0.97333

    def test_window_limit_on_length_average(self):
        nz = noise(self.N_KL, self.N_T1)
        # 8192: positions 3.4 N / 2.0 N / 2.0 N -> average 2.47 N <= 3 N passes (the old per-window rule failed p0)
        res = [win(1, 0.0068, .97), win(2, 0.004, .97), win(3, 0.004, .97)] + [win(4, 0.003, .97)] * 3
        cs = klgate.judge(res, GROUPS, POSS, "balance", True, nz)
        ln = by_rule(cs, "moe-length")
        self.assertEqual(len(ln), 2)
        self.assertAlmostEqual(ln[0]["x_noise"], 0.0148 / 3 / 0.002)
        self.assertTrue(ln[0]["pass"])
        self.assertAlmostEqual(ln[0]["kl_limit"], 0.006)
        # 8192 average 3.05 N > 3 N fails
        res[1] = win(2, 0.0075, .97)
        cs = klgate.judge(res, GROUPS, POSS, "balance", True, nz)
        self.assertFalse(by_rule(cs, "moe-length")[0]["pass"])

    def test_top1_floor_is_all_window_noise_mean(self):
        nz = noise(self.N_KL, self.N_T1)
        allmean = sum(self.N_T1) / 6
        # 8192 average top-1 0.945 < all-window floor 0.97667 - 0.030 = 0.94667 -> fail
        # (the 8192-only noise mean 0.97333 - 0.030 = 0.94333 would have passed)
        res = [win(1, 0.002, .945)] * 3 + [win(4, 0.002, .97)] * 3
        ln = by_rule(klgate.judge(res, GROUPS, POSS, "balance", True, nz), "moe-length")
        self.assertAlmostEqual(ln[0]["noise_top1"], allmean)
        self.assertAlmostEqual(ln[0]["top1_limit"], allmean - 0.030)
        self.assertAlmostEqual(ln[1]["top1_limit"], allmean - 0.030)  # the same floor for every length
        self.assertFalse(ln[0]["pass"])
        self.assertTrue(ln[1]["pass"])
        # the reverse: a length whose own noise top-1 is high does not get a stricter floor
        nz2 = noise(self.N_KL, [0.999, 0.999, 0.999, 0.95, 0.95, 0.95])  # all-window mean 0.9745
        res = [win(1, 0.002, .955)] * 3 + [win(4, 0.002, .955)] * 3   # an 8192 same-window floor would be 0.975
        ln = by_rule(klgate.judge(res, GROUPS, POSS, "balance", True, nz2), "moe-length")
        self.assertTrue(all(c["pass"] for c in ln))

    def test_top1_boundary(self):
        nz = noise([0.002] * 3, [0.97] * 3)
        at = klgate.judge([win(1, 0.002, 0.97 - 0.030)] * 3, GROUPS[:3], POSS[:3], "balance", True, nz)
        under = klgate.judge([win(1, 0.002, 0.97 - 0.0301)] * 3, GROUPS[:3], POSS[:3], "balance", True, nz)
        self.assertTrue(by_rule(at, "moe-length")[0]["pass"])
        self.assertFalse(by_rule(under, "moe-length")[0]["pass"])

    def test_balance_top1_drop_is_3_points(self):
        # MOE-BAL-3 (owner decision 2026-10-14 23:36): MoE balance top-1 drop 2.4 -> 3.0 points; the KL
        # multipliers (2 N mean, 3 N per length), fast and the dense absolute limits are unchanged
        self.assertEqual(klgate.MOE_LIMITS["balance"], (2.0, 3.0, 0.030))
        self.assertEqual(klgate.MOE_LIMITS["fast"], (8.0, 12.0, 0.08))
        self.assertEqual(klgate.LIMITS, {"balance": (0.005, 0.98), "fast": (0.03, 0.93)})
        # the measured case that failed the 2.4-point floor: R9700 MoE zh 8k with q8dec, 93.88% vs the
        # noise all-window mean 96.40% (-2.52 points), 2.35 N -> passes now
        nz = noise([0.00597] * 3, [0.9640] * 3)
        cs = klgate.judge([win(1, 0.01400, 0.9388)] * 3, GROUPS[:3], POSS[:3], "balance", True, nz)
        ln = by_rule(cs, "moe-length")[0]
        self.assertTrue(ln["pass"])
        self.assertAlmostEqual(ln["top1_limit"], 0.9340)
        # -3.1 points still fails; KL over 3 N still fails whatever the top-1
        self.assertFalse(by_rule(klgate.judge([win(1, 0.002, 0.9330)] * 3, GROUPS[:3], POSS[:3], "balance", True, nz), "moe-length")[0]["pass"])
        self.assertFalse(by_rule(klgate.judge([win(1, 0.0180, 0.9640)] * 3, GROUPS[:3], POSS[:3], "balance", True, nz), "moe-length")[0]["pass"])

    def test_kl_boundary(self):
        nz = noise([0.002] * 3, [0.97] * 3)
        at = klgate.judge([win(1, 0.006, 0.97)] * 3, GROUPS[:3], POSS[:3], "balance", True, nz)  # exactly 3 N
        over = klgate.judge([win(1, 0.00601, 0.97)] * 3, GROUPS[:3], POSS[:3], "balance", True, nz)
        self.assertTrue(by_rule(at, "moe-length")[0]["pass"])
        self.assertFalse(by_rule(over, "moe-length")[0]["pass"])

    def test_mean_rule_over_all_windows(self):
        nz = noise(self.N_KL, self.N_T1)  # N = 0.002, balance mean limit 0.004
        # every length average <= 3 N, but the mean over all windows 2.05 N > 2 N: only the mean case fails
        res = [win(1, 0.0041, .98)] * 6
        cs = klgate.judge(res, GROUPS, POSS, "balance", True, nz)
        m = by_rule(cs, "moe-mean")
        self.assertEqual(len(m), 1)
        self.assertFalse(m[0]["pass"])
        self.assertAlmostEqual(m[0]["x_noise"], 2.05)
        self.assertTrue(all(c["pass"] for c in by_rule(cs, "moe-length")))
        # exactly 2 N passes
        res = [win(1, 0.004, .98)] * 6
        self.assertTrue(by_rule(klgate.judge(res, GROUPS, POSS, "balance", True, nz), "moe-mean")[0]["pass"])

    def test_mean_is_over_windows_not_lengths(self):
        # unequal window counts per length: the mean is over windows, the per-length rule over that length's windows
        nz = noise([0.002] * 4, [0.98] * 4)
        res = [win(1, 0.001, .98), win(2, 0.001, .98), win(3, 0.001, .98), win(4, 0.007, .98)]
        cs = klgate.judge(res, [8192, 8192, 8192, 32768], [0, 1, 2, 0], "balance", True, nz)
        self.assertAlmostEqual(by_rule(cs, "moe-mean")[0]["kl_mean"], 0.0025)  # not (0.001 + 0.007) / 2
        self.assertFalse(by_rule(cs, "moe-length")[1]["pass"])  # 3.5 N

    def test_fast_limits(self):
        nz = noise([0.002] * 3, [0.97] * 3)
        cs = klgate.judge([win(1, 0.024, .89)] * 3, GROUPS[:3], POSS[:3], "fast", True, nz)  # 12 N, 0.97 - 0.08
        self.assertTrue(by_rule(cs, "moe-length")[0]["pass"])
        self.assertFalse(by_rule(cs, "moe-mean")[0]["pass"])  # 12 N > 8 N

    def test_case_lines_show_positions_and_limits(self):
        nz = noise(self.N_KL, self.N_T1)
        res = [win(11, 0.0068, .97), win(12, 0.004, .97), win(13, 0.004, .97)] + [win(4, 0.003, .97)] * 3
        cs = klgate.judge(res, GROUPS, POSS, "balance", True, nz)
        txt = "\n".join(l for c in cs for l in klgate.case_lines("moe", "code", "balance", c, nz))
        for want in ("8192/p0 (end 11)", "3.40 N", "avg of 3 positions", "<= 0.00600 (3 N)", "noise top1 all-window mean 97.67%",
                     "mean over all windows", "(2 N)", "not judged alone"):
            self.assertIn(want, txt)


class Replay(unittest.TestCase):
    """Re-judge a stored KL-ROBUST (3-position, per-window) report with the new rule, if present (prints only)."""
    P = os.environ.get("KLGATE_REPLAY", r"C:\ClaudeWorkspace\whirl-work\klrobust\p3_r9700.json")

    def test_replay(self):
        if not os.path.isfile(self.P):
            self.skipTest("no stored report")
        with open(self.P, encoding="utf-8") as f:
            rep = json.load(f)
        noise_by = {(n["model"], n["prompt"]): n for n in rep["noise"]}
        keys = []
        for c in rep["cases"]:
            k = (c["model"], c["prompt"], c["mode"])
            if c["rule"] != "moe-mean" and k not in keys:
                keys.append(k)
        out = []
        for m, p, md in keys:
            ws = [c for c in rep["cases"] if (c["model"], c["prompt"], c["mode"]) == (m, p, md) and c["rule"] != "moe-mean"]
            res = [(c["length"], c["zero_rows"], c["kl_mean"], c["kl_p99"], c["kl_max"], c["top1"]) for c in ws]
            nz = noise_by.get((m, p))
            cs = klgate.judge(res, [c["group"] for c in ws], [c["pos"] for c in ws], md, nz is not None, nz)
            out += [(m, p, md, c) for c in cs]
        print()
        for m, p, md, c in out:
            print(f"  replay {m}:{p}:{md} {c['rule']} {c['length']}: KL {c['kl_mean']:.5f} <= {c['kl_limit']:.5f}"
                  + (f" top1 {c['top1'] * 100:.2f}% >= {c['top1_limit'] * 100:.2f}%" if "top1" in c else "")
                  + f" {'PASS' if c['pass'] else 'FAIL'}")
        self.assertTrue(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
