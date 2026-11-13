"""合成面板生成与核对套件。

核对项（对应用户要求）：
1. 已知恒定处理效应的合成面板：逐格与事件研究 ATT 应恢复真值；
2. 缺支持/完全分离：显式不可估并带原因，不产出伪估计；
3. 地区标签重排：折分随地区同映射后，点估计与权重数值不变；
4. 事件窗缩短：共同事件期结果不变，差异只能来自参与队列集合；
另含提前期使晚处理地区退出对照的专门核对。
"""
import random
from fractions import Fraction

from .cells import estimate_cell
from .estimator import _run_from_analysis
from .panel import Analysis

TRUE_EFFECT = Fraction(7)
YEARS = (0, 6)
HH_PER_DISTRICT = 2


def _make_analysis(rng, districts, cohort_map, separated=False, L=0,
                   seed=0, replicates=199, window=(0, 1), folds=None,
                   heterogeneous=False, continuous_x=False):
    """构造内存 Analysis（形状与 panel.load_analysis 输出一致）。

    DGP：平行趋势 Y0_t = 100 + 2t（可选地区异质水平），处理后加恒定效应；
    协变量在家庭层面固定（模拟基期登记），全程不变，符合“协变量取b时点”。
    continuous_x=True 时用连续协变量，保证折内设计通常满秩；
    separated=True 时用队列完全分离的二值协变量。
    正确设定的 m0（截距+x）与 p 下，DR 应精确恢复处理效应（ΔY 口径）。
    """
    households, district_of, district_hh, parsed, weights = {}, {}, {}, {}, {}
    rows = []
    cov_names = ["x"]
    if folds is None:
        # 折号附着在“地区实体”上：按队列内出现顺序循环分配，
        # 与地区名字典序无关，保证纯重命名后分组结构完全不变。
        folds = {}
        per_cohort_counter = {}
        for d in districts:
            g = cohort_map[d]
            idx = per_cohort_counter.get(g, 0)
            per_cohort_counter[g] = idx + 1
            folds[d] = idx % 3
    for d in districts:
        g = cohort_map[d]
        district_hh[d] = set()
        base_level = rng.uniform(90, 110) if heterogeneous else 100.0
        for k in range(HH_PER_DISTRICT):
            hh = f"{d}-h{k}"
            district_of[hh] = d
            district_hh[d].add(hh)
            weights[hh] = Fraction(1)
            households[hh] = {}
            if separated:
                x_fixed = 1 if g == 3 else 0
            elif continuous_x:
                x_fixed = round(rng.uniform(-1, 1), 4)
            else:
                x_fixed = rng.randrange(2)
            for t in range(YEARS[0], YEARS[1] + 1):
                treated = g is not None and t >= g
                y0 = base_level + 2.0 * t
                y = y0 + (float(TRUE_EFFECT) if treated else 0.0)
                parsed.setdefault(hh, {})[t] = {
                    "x": (Fraction(str(x_fixed)),),
                    "outcome": Fraction(str(round(y, 6)))}
                households[hh][t] = {"household": hh, "district": d,
                                     "year": t, "G": g, "x": str(x_fixed),
                                     "outcome": str(round(y, 6))}
                rows.append(households[hh][t])
    cfg = {
        "anticipation": L, "event_window": window, "folds": folds,
        "covariates": ["x"],
        "weights_raw": {hh: "1" for hh in households},
        "bootstrap_seed": seed, "bootstrap_replicates": replicates,
        "multiplier": "Rademacher", "confidence": 19 / 20,
        "tolerance": 1e-8,
    }
    cohorts = tuple(sorted({g for g in cohort_map.values() if g is not None}))
    never = tuple(sorted(d for d, g in cohort_map.items() if g is None))
    return Analysis(
        rows=rows, years=YEARS, households=households, district_of=district_of,
        district_g=dict(cohort_map), district_hh=district_hh, folds=dict(folds),
        weights=weights, parsed=parsed, cohorts=cohorts, never=never, cfg=cfg)


def _relabel_analysis(an, remap):
    """对同一份数据做地区重命名（家庭名同步），折分随地区映射，数据值不变。"""
    district_g = {remap[d]: g for d, g in an.district_g.items()}
    folds = {remap[d]: f for d, f in an.folds.items()}
    district_of = {hh: remap[d] for hh, d in an.district_of.items()}
    hh_remap = {hh: f"{remap[an.district_of[hh]]}-{hh.split('-', 1)[1]}"
                for hh in an.households}
    households, parsed, weights, district_hh = {}, {}, {}, {}
    rows = []
    for hh, panel in an.households.items():
        nh = hh_remap[hh]
        households[nh] = {}
        parsed[nh] = an.parsed[hh]
        weights[nh] = an.weights[hh]
        d2 = district_of[hh]
        district_hh.setdefault(d2, set()).add(nh)
        for t, r in panel.items():
            nr = dict(r, household=nh, district=d2)
            households[nh][t] = nr
            rows.append(nr)
    cfg = dict(an.cfg, folds=folds,
               weights_raw={hh_remap[h]: "1" for h in an.households})
    return Analysis(
        rows=rows, years=an.years, households=households,
        district_of=district_of, district_g=district_g,
        district_hh=district_hh, folds=folds, weights=weights,
        parsed=parsed, cohorts=an.cohorts, never=tuple(sorted(
            remap[d] for d in an.never)), cfg=cfg)


def _default_cohort_map():
    districts = [f"d{i}" for i in range(12)]
    cohort_map = {}
    for i, d in enumerate(districts):
        cohort_map[d] = 3 if i < 4 else (4 if i < 8 else None)
    return districts, cohort_map


def _clone_with_window(an, window):
    """同数据克隆，仅替换事件窗配置。"""
    cfg = dict(an.cfg, event_window=window)
    return Analysis(
        rows=an.rows, years=an.years, households=an.households,
        district_of=an.district_of, district_g=an.district_g,
        district_hh=an.district_hh, folds=an.folds, weights=an.weights,
        parsed=an.parsed, cohorts=an.cohorts, never=an.never, cfg=cfg)


def run_validation_suite(seed=20261113, replicates=199):
    rng = random.Random(seed)
    report = {"true_effect": float(TRUE_EFFECT), "checks": []}

    # --- 1. 恒定效应恢复 -----------------------------------------------------
    districts, cohort_map = _default_cohort_map()
    an = _make_analysis(rng, districts, cohort_map, seed=seed,
                        replicates=replicates)
    pkg = _run_from_analysis(an, trim=None, n_placebo=None,
                             do_bootstrap=True, do_placebo=True)
    cell_checks = []
    for r in pkg["cells"]:
        if r["estimable"]:
            err = abs(r["att"] - float(TRUE_EFFECT))
            cell_checks.append({"g": r["g"], "t": r["t"], "att": r["att"],
                                "abs_error": err,
                                "pass": err < 1e-6})
        else:
            cell_checks.append({"g": r["g"], "t": r["t"], "estimable": False,
                                "reason": r["reason"], "pass": False})
    event_checks = [{
        "event_time": r["event_time"],
        "att": r.get("att"),
        "abs_error": abs(r["att"] - float(TRUE_EFFECT)) if r["estimable"] else None,
        "pass": r["estimable"] and abs(r["att"] - float(TRUE_EFFECT)) < 1e-6,
    } for r in pkg["event_study"]["rows"]]
    band = pkg["bootstrap"]
    band_covers = all(
        (r["ci_low"] <= float(TRUE_EFFECT) <= r["ci_high"])
        for r in band["rows"] if r["estimable"])
    placebo_rows = [r for r in pkg.get("placebo", {}).get(
        "event_study", {}).get("rows", []) if r["estimable"]]
    placebo_zero = all(abs(r["att"]) < 1e-6 for r in placebo_rows)
    report["checks"].append({
        "name": "constant_effect_recovery",
        "pass": all(c["pass"] for c in cell_checks)
        and all(c["pass"] for c in event_checks),
        "cells": cell_checks, "event_study": event_checks,
        "simultaneous_band_covers_truth": band_covers,
        "band_published": band["publish_band"],
        "placebo_att_all_zero": placebo_zero,
        "note": "合成真值为恒定+7；安慰剂为诊断，零值不证明平行趋势。",
    })

    # --- 2. 缺支持/完全分离 --------------------------------------------------
    an_sep = _make_analysis(random.Random(seed + 1), districts, cohort_map,
                            separated=True, seed=seed, replicates=replicates)
    sep_cells = []
    valid_stages = ("cell_construction", "no_common_support", "fold_support")
    for g in an_sep.cohorts:
        for t in range(g, YEARS[1] + 1):
            r = estimate_cell(an_sep, g, t, kind="post")
            stage_ok = (r["stage"] in valid_stages
                        or (isinstance(r["stage"], str)
                            and (r["stage"].startswith("logit_fold")
                                 or r["stage"].startswith("ols_fold"))))
            sep_cells.append({
                "g": g, "t": t, "estimable": r["estimable"],
                "reason": r["reason"], "stage": r["stage"],
                "pass": (not r["estimable"]) and stage_ok})
    report["checks"].append({
        "name": "lack_of_support_separation",
        "pass": all(c["pass"] for c in sep_cells),
        "cells": sep_cells,
        "note": "g=3 队列协变量与对照完全分离，必须显式不可估，不得正则化掩盖。",
    })

    # --- 3. 地区标签重排：等结构重命名，估计与权重逐格不变 -------------------
    # 隔离“名字”维度：仅在同队列且同折号的地区之间互换ID，
    # 折的实体成员、G归属、数据全部不动，估计必须逐格完全一致。
    groups = {}
    for d in districts:
        key = (cohort_map[d], an.folds[d])
        groups.setdefault(key, []).append(d)
    remap = {}
    rng_lab = random.Random(seed + 2)
    for members in groups.values():
        shuffled = members[:]
        rng_lab.shuffle(shuffled)
        remap.update(dict(zip(members, shuffled)))
    an2 = _relabel_analysis(an, remap)
    pkg2 = _run_from_analysis(an2, trim=None, n_placebo=None,
                              do_bootstrap=False, do_placebo=False)
    relabel = []
    by_key = {(r["g"], r["t"]): r for r in pkg2["cells"]}
    for r1 in pkg["cells"]:
        r2 = by_key[(r1["g"], r1["t"])]
        ok = (r1["estimable"] == r2["estimable"]
              and ((not r1["estimable"]) or abs(r1["att"] - r2["att"]) < 1e-12))
        relabel.append({"g": r1["g"], "t": r1["t"],
                        "att_before": r1["att"], "att_after": r2["att"],
                        "pass": ok})
    report["checks"].append({
        "name": "district_label_permutation_invariance",
        "pass": all(c["pass"] for c in relabel),
        "cells": relabel,
        "note": "仅重命名地区且折分随地区同映射，估计与权重必须逐格一致。",
    })

    # --- 4. 事件窗缩短：共同事件期不变 --------------------------------------
    an_short = _clone_with_window(an, (0, 0))
    short_pkg = _run_from_analysis(an_short, trim=None, n_placebo=None,
                                   do_bootstrap=False, do_placebo=False)
    full_rows = {r["event_time"]: r for r in pkg["event_study"]["rows"]}
    short_rows = {r["event_time"]: r for r in short_pkg["event_study"]["rows"]}
    common = sorted(set(full_rows) & set(short_rows))
    shorten = [{
        "event_time": e,
        "att_full_window": full_rows[e].get("att"),
        "att_short_window": short_rows[e].get("att"),
        "pass": abs(full_rows[e]["att"] - short_rows[e]["att"]) < 1e-12,
    } for e in common]
    report["checks"].append({
        "name": "event_window_shortening_stability",
        "pass": all(c["pass"] for c in shorten),
        "common_event_times": shorten,
        "full_participating": pkg["event_study"]["participating_cohorts"],
        "short_participating": short_pkg["event_study"]["participating_cohorts"],
    })

    # --- 5. 提前期 L=1：晚处理地区在 t=g4-L-? 退出对照 ----------------------
    an_L = _make_analysis(random.Random(seed + 5), districts, cohort_map,
                          L=1, seed=seed, replicates=replicates)
    exit_rows = []
    for row in _run_from_analysis(
            an_L, trim=None, n_placebo=None, do_bootstrap=False,
            do_placebo=False)["control_exit_schedule"]:
        if row["g"] == 3:
            exit_rows.append({
                "t": row["t"],
                "legal": row["legal_control_districts"],
                "g4_status": next(d["status"] for d in row["district_status"]
                                  if d["district"].startswith("d4")),
            })
    l1 = next(r for r in exit_rows if r["t"] == 3)
    report["checks"].append({
        "name": "anticipation_removes_later_cohort_controls",
        "pass": l1["g4_status"] == "excluded_in_anticipation"
        and all(not d.startswith("d4") for d in l1["legal"]),
        "schedule_g3": exit_rows,
        "note": "L=1 时 G=4 地区在 t=3 已进入提前期，必须退出 g=3 的对照集。",
    })

    # --- 6. 显式修剪改变目标时单独报告 --------------------------------------
    trim_pkg = _run_from_analysis(an, trim=0.1, n_placebo=None,
                                  do_bootstrap=False, do_placebo=False)
    trim_reports = [r["trim_report"] for r in trim_pkg["cells"]
                    if r["estimable"] and r["trim_report"]]
    report["checks"].append({
        "name": "explicit_trim_reports_new_target",
        "pass": all("excluded_households" in tr for tr in trim_reports),
        "n_cells_with_trim": len(trim_reports),
        "estimand_note": trim_pkg["config"]["estimand"],
    })

    # --- 7. 充足聚类的大面板：同时带可发布且覆盖真值 -------------------------
    # 小面板（每队列4地区）下0/2乘子必然频繁整折失效、应拒发区间；
    # 这里构造每队列15个处理/对照地区，验证推断机制本身有效。
    big_districts = [f"b{i}" for i in range(45)]
    big_map = {d: (3 if i < 15 else 4 if i < 30 else None)
               for i, d in enumerate(big_districts)}
    an_big = _make_analysis(random.Random(seed + 7), big_districts, big_map,
                            seed=seed, replicates=replicates, continuous_x=True)
    big_pkg = _run_from_analysis(an_big, trim=None, n_placebo=None,
                                 do_bootstrap=True, do_placebo=True)
    bb = big_pkg["bootstrap"]
    band_ok = bb["publish_band"] and all(
        r["estimable"] and r["ci_low"] <= float(TRUE_EFFECT) <= r["ci_high"]
        for r in bb["rows"])
    point_ok = all(
        abs(r["att"] - float(TRUE_EFFECT)) < 1e-6
        for r in big_pkg["event_study"]["rows"])
    report["checks"].append({
        "name": "large_cluster_panel_simultaneous_band",
        "pass": band_ok and point_ok
        and bb["n_success"] / bb["replicates"] >= 0.9,
        "n_success": bb["n_success"], "replicates": bb["replicates"],
        "publish_band": bb["publish_band"],
        "failure_reasons": bb["failure_reasons"],
        "band_rows": [{"event_time": r["event_time"],
                       "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"),
                       "att": r.get("att")} for r in bb["rows"]],
        "note": "聚类充足时同时带应发布并覆盖真值；与小面板的拒发行为互为对照。",
    })

    # --- 8. 小面板必须诚实拒发区间，不输出伪精确 -----------------------------
    small_boot = pkg["bootstrap"]
    report["checks"].append({
        "name": "small_cluster_refuses_pseudo_precision",
        "pass": (not small_boot["publish_band"])
        and small_boot["n_success"] < small_boot["replicates"]
        and bool(small_boot["failure_reasons"]),
        "n_success": small_boot["n_success"],
        "failure_reasons": small_boot["failure_reasons"],
        "note": "Rademacher 0/2下整折处理地区被置零即不可估；"
                "有效次数不足时只给点估计与原因，不发布置信带。",
    })

    report["all_pass"] = all(c["pass"] for c in report["checks"])
    return report
