"""分期ATT分析模块的口径测试（纯标准库）。"""
import random
import unittest

from analysis_data.cells import estimate_cell
from analysis_data.estimator import run_analysis
from analysis_data.numerics import FitFailure, exact_rank, weighted_logit, weighted_ols
from analysis_data.panel import PanelError, load_analysis
from analysis_data.synthetic import (
    TRUE_EFFECT,
    _default_cohort_map,
    _make_analysis,
    run_validation_suite,
)


class NumericsTests(unittest.TestCase):
    def test_exact_rank_detects_collinearity(self):
        self.assertEqual(exact_rank([[1, 2], [2, 4], [1, 1]]), 2)
        self.assertEqual(exact_rank([[1, 1], [1, 1]]), 1)
        self.assertEqual(exact_rank([[0, 0]]), 0)

    def test_ols_rank_failure_explicit(self):
        with self.assertRaises(FitFailure) as ctx:
            weighted_ols([[1], [1], [1]], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        self.assertEqual(ctx.exception.code, "rank_deficient")

    def test_logit_separation_detected(self):
        # 完全分离：x 完美预测 y
        X = [[0.0], [0.0], [1.0], [1.0]]
        y = [0, 0, 1, 1]
        with self.assertRaises(FitFailure) as ctx:
            weighted_logit(X, y, [1.0] * 4, tol=1e-8, max_iter=200)
        self.assertEqual(ctx.exception.code, "separated")

    def test_logit_single_class(self):
        with self.assertRaises(FitFailure) as ctx:
            weighted_logit([[0.0], [1.0]], [1, 1], [1.0, 1.0])
        self.assertEqual(ctx.exception.code, "separated")


class CellContractTests(unittest.TestCase):
    def setUp(self):
        districts, cm = _default_cohort_map()
        self.an = _make_analysis(random.Random(7), districts, cm,
                                 seed=7, replicates=99)

    def test_legal_controls_exclude_treated_and_anticipating(self):
        # L=0: t=3 时 G=4 合法；t=4 起 G=4 已处理，退出对照
        c3 = estimate_cell(self.an, 3, 3)
        c4 = estimate_cell(self.an, 3, 4)
        self.assertTrue(any(d.startswith("d4") for d in c3["control_districts"]))
        self.assertFalse(any(d.startswith("d4") for d in c4["control_districts"]))
        self.assertTrue(all(d.startswith(("d8", "d9", "d10", "d11"))
                            for d in c4["control_districts"]))

    def test_fixed_base_year(self):
        c = estimate_cell(self.an, 3, 4)
        self.assertEqual(c["b"], 3 - 0 - 1)

    def test_no_household_both_train_and_eval_fold(self):
        c = estimate_cell(self.an, 3, 3)
        # 折按地区固定：同地区家庭只能在一个折，训练折不含评估地区
        for diag in c["fold_diagnostics"]:
            self.assertGreaterEqual(diag["n_train_treated"], 1)
            self.assertGreaterEqual(diag["n_train_control"], 1)

    def test_recovers_constant_effect_on_change_outcome(self):
        for g in self.an.cohorts:
            for t in range(g, self.an.year_max + 1):
                r = estimate_cell(self.an, g, t)
                self.assertTrue(r["estimable"], (g, t, r["reason"]))
                self.assertAlmostEqual(r["att"], float(TRUE_EFFECT), places=8)

    def test_weights_normalized_within_cell(self):
        r = estimate_cell(self.an, 3, 3)
        q = sum(item["weight"] for item in
                r["weight_diagnostics"]["control_normalized_ipw"])
        self.assertAlmostEqual(q, 1.0, places=10)
        qt = sum(item["weight"] for item in
                 r["weight_diagnostics"]["treated_normalized_weights"])
        self.assertAlmostEqual(qt, 1.0, places=10)


class SeparatedPanelTests(unittest.TestCase):
    def test_separated_panel_explicitly_not_estimable(self):
        districts, cm = _default_cohort_map()
        an = _make_analysis(random.Random(8), districts, cm,
                            separated=True, seed=8)
        for g in an.cohorts:
            for t in range(g, an.year_max + 1):
                r = estimate_cell(an, g, t)
                self.assertFalse(r["estimable"])
                self.assertTrue(r["stage"] in (
                    "cell_construction", "no_common_support", "fold_support")
                    or r["stage"].startswith(("logit_fold", "ols_fold")))

    def test_real_separated_manifest_not_estimable(self):
        pkg = run_analysis("data/separated-propensity.json",
                           root="data", do_bootstrap=False, do_placebo=False)
        self.assertTrue(all(not c["estimable"] for c in pkg["cells"]))


class AnticipationTests(unittest.TestCase):
    def test_anticipation_removes_controls(self):
        pkg = run_analysis("data/anticipation-removes-controls.json",
                           root="data", do_bootstrap=False, do_placebo=False)
        self.assertEqual(pkg["config"]["anticipation_L"], 1)
        sched = next(r for r in pkg["control_exit_schedule"]
                     if r["g"] == 3 and r["t"] == 3)
        self.assertNotIn("d4", sched["legal_control_districts"])
        status = next(d["status"] for d in sched["district_status"]
                      if d["district"] == "d4")
        self.assertEqual(status, "excluded_in_anticipation")
        # 基期前移到 g-L-1
        cell = next(c for c in pkg["cells"] if c["g"] == 3 and c["t"] == 3)
        self.assertEqual(cell["b"], 1)


class ValidationSuiteTests(unittest.TestCase):
    def test_full_suite_passes(self):
        report = run_validation_suite(seed=20261113, replicates=99)
        self.assertTrue(report["all_pass"])
        names = {c["name"] for c in report["checks"]}
        self.assertIn("constant_effect_recovery", names)
        self.assertIn("small_cluster_refuses_pseudo_precision", names)
        self.assertIn("large_cluster_panel_simultaneous_band", names)


class ManifestValidationTests(unittest.TestCase):
    def test_negative_anticipation_rejected(self):
        with self.assertRaises((PanelError, ValueError)):
            load_analysis("data/invalid-config.json", root="data")

    def test_normal_manifest_loads(self):
        an = load_analysis("data/normal.json", root="data")
        self.assertEqual(an.L, 0)
        self.assertEqual(an.cohorts, (3, 4))
        self.assertEqual(len(an.never), 4)


if __name__ == "__main__":
    unittest.main()
