"""地区聚类乘子重抽样与同时置信带。

口径（docs/data-contract.md）：
- 乘子按 district 生成，同地区全户同乘；每次重抽重做全部格拟合；
- 随机种子与折分在重抽间固定（折分本来就由清单固定，不重划）；
- 固定原分析的完整事件窗可估队列与基期人口权重；
- 以 bootstrap 居中标准化事件向量最大绝对值分位数构造同时带；
- 退化坐标单独标记；不可估次数与原因单列，有效次数不足不发布区间。
"""
import random

from .cells import estimate_cell
from .numerics import FitFailure, quantile_nearest_rank, safe_sqrt

MIN_SUCCESS_FRACTION = 0.9   # 有效重抽比例下限
MIN_SUCCESS_ABSOLUTE = 50    # 有效重抽绝对次数下限


def _draw_multipliers(rng, districts, kind):
    if kind == "Rademacher":
        # 0/2 权重 = 1 + Rademacher，与契约一致
        return {d: 2 * rng.randrange(2) for d in districts}
    if kind == "Mammen":
        # Mammen 两点分布：(1-sqrt5)/2 概率 (sqrt5+1)/(2sqrt5)，否则 (1+sqrt5)/2
        phi = (1 + 5 ** 0.5) / 2
        low, high = 1 - phi, phi
        p_low = high / (phi + (1 / phi))  # = (sqrt5+1)/(2sqrt5)
        return {d: (low if rng.random() < p_low else high) for d in districts}
    raise ValueError(f"未知乘子类型 {kind}")


def run_bootstrap(an, grid, event_window, event_study_fn, kind,
                  seed=None, replicates=None, multiplier=None,
                  confidence=None, trim=None, fixed_cohorts=None,
                  fixed_weights=None):
    """对事件研究聚合向量做地区乘子 bootstrap。

    event_study_fn: 以 cell_results 为输入返回聚合 dict 的闭包；
      bootstrap 闭包注入 fixed_cohorts/fixed_weights，重抽不改变聚合目标。
    fixed_cohorts/fixed_weights: 原分析确定的参与队列与基期人口权重。
    返回同时带、点wise SE、有效次数与不可估明细。
    """
    seed = an.cfg["bootstrap_seed"] if seed is None else seed
    replicates = an.cfg["bootstrap_replicates"] if replicates is None else replicates
    multiplier = an.cfg["multiplier"] if multiplier is None else multiplier
    confidence = an.cfg["confidence"] if confidence is None else confidence
    districts = sorted(an.district_g)
    treated_districts = [d for d in districts if an.district_g[d] is not None]
    rng = random.Random(seed)

    def _fit_grid(mult=None):
        out = []
        for item in grid:
            if len(item) == 3:
                g, t, base = item
                out.append(estimate_cell(an, g, t, kind=kind, trim=trim,
                                          multipliers=mult, base_override=base))
            else:
                g, t = item
                out.append(estimate_cell(an, g, t, kind=kind, trim=trim,
                                          multipliers=mult))
        return out

    def _agg(cells):
        return event_study_fn(cells, fixed_cohorts, fixed_weights)

    base_agg = _agg(_fit_grid(None))
    event_points = [r["event_time"] for r in base_agg["rows"] if r["estimable"]]
    base_atts = {r["event_time"]: r["att"] for r in base_agg["rows"] if r["estimable"]}

    # 原样本即无任何可估事件点：bootstrap 无目标可推断，明确不发布而非计为成功
    if not event_points:
        reasons = _tally([
            {"stage": "point_estimate", "reason": r.get("reason", "not_estimable")}
            for r in base_agg["rows"] if not r["estimable"]])
        return {
            "kind": kind, "seed": seed, "replicates": replicates,
            "multiplier": multiplier, "n_success": 0,
            "n_failures": replicates, "failure_reasons": reasons,
            "failures": [], "zero_weight_treated_replicates": 0,
            "n_treated_clusters": len(treated_districts),
            "cluster_warning": None, "publish_band": False,
            "min_success_rule": {"fraction": MIN_SUCCESS_FRACTION,
                                 "absolute": MIN_SUCCESS_ABSOLUTE},
            "rows": [{"event_time": r["event_time"], "estimable": False,
                      "reason": r.get("reason", "not_estimable"),
                      "detail": "原样本格估计不可估，bootstrap无目标可推断"}
                     for r in base_agg["rows"]],
            "diagnostic_only": kind == "placebo",
        }

    boot_vectors = []
    failures = []
    zero_weight_reps = 0
    for rep in range(replicates):
        mult = _draw_multipliers(rng, districts, multiplier)
        # 0/2 下若处理地区全为0，该重抽对目标无信息，单独计数
        if all(mult[d] == 0 for d in treated_districts):
            zero_weight_reps += 1
        try:
            cells = _fit_grid(mult)
            agg = _agg(cells)
            vec = {}
            ok = True
            for e in event_points:
                row = next((r for r in agg["rows"] if r["event_time"] == e), None)
                if row is None or not row.get("estimable"):
                    ok = False
                    failures.append({
                        "replicate": rep, "event_time": e,
                        "reason": (row or {}).get("reason", "missing_event_point"),
                        "stage": "aggregation"})
                    break
                vec[e] = row["att"]
            if ok:
                boot_vectors.append(vec)
        except FitFailure as f:
            failures.append({"replicate": rep, "reason": f.code,
                             "detail": f.detail, "stage": "fit"})
        except Exception as f:  # 防御：任何意外都登记而非静默
            failures.append({"replicate": rep, "reason": type(f).__name__,
                             "detail": str(f), "stage": "unexpected"})

    n_success = len(boot_vectors)
    n_treated_clusters = len(treated_districts)
    cluster_warning = None
    if n_treated_clusters <= 10:
        cluster_warning = (f"处理地区仅{n_treated_clusters}个，聚类渐近近似可能不可靠；"
                           "Rademacher乘子的有效重抽分布本身也受聚类数限制")

    publish = (n_success >= MIN_SUCCESS_ABSOLUTE
               and n_success / replicates >= MIN_SUCCESS_FRACTION)

    rows = []
    max_stats = []
    pointwise = {}
    for e in event_points:
        vals = [v[e] for v in boot_vectors]
        if len(vals) < 2:
            rows.append({"event_time": e, "estimable": False,
                         "reason": "insufficient_bootstrap_replicates"})
            continue
        mu = sum(vals) / len(vals)
        se = safe_sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
        pointwise[e] = se
        if se == 0.0:
            rows.append({"event_time": e, "estimable": False, "att": base_atts[e],
                         "reason": "degenerate_variance",
                         "detail": "bootstrap方差为0，不构造伪精确区间"})
            continue
        max_stats.extend(abs((v - mu) / se) for v in vals)
    # 同时带：所有非退化坐标共用一个 max-t 临界值
    crit = (quantile_nearest_rank(max_stats, confidence)
            if publish and max_stats else None)
    for e in event_points:
        row = next(r for r in rows if r["event_time"] == e) if any(
            r["event_time"] == e for r in rows) else None
        if row is not None and not row["estimable"]:
            continue
        se = pointwise.get(e)
        if crit is None or se is None:
            rows.append({"event_time": e, "estimable": False, "att": base_atts[e],
                         "reason": "band_not_published",
                         "detail": "有效重抽不足或坐标退化"})
            continue
        rows.append({
            "event_time": e, "estimable": True, "att": base_atts[e],
            "bootstrap_mean": sum(v[e] for v in boot_vectors) / n_success,
            "pointwise_se": se,
            "simultaneous_critical_value": crit,
            "ci_low": base_atts[e] - crit * se,
            "ci_high": base_atts[e] + crit * se,
            "confidence": confidence,
        })
    return {
        "kind": kind,
        "seed": seed, "replicates": replicates, "multiplier": multiplier,
        "n_success": n_success,
        "n_failures": replicates - n_success,
        "failure_reasons": _tally(failures),
        "failures": failures[:50],
        "zero_weight_treated_replicates": zero_weight_reps,
        "n_treated_clusters": n_treated_clusters,
        "cluster_warning": cluster_warning,
        "publish_band": bool(publish and crit is not None),
        "min_success_rule": {"fraction": MIN_SUCCESS_FRACTION,
                             "absolute": MIN_SUCCESS_ABSOLUTE},
        "rows": sorted(rows, key=lambda r: r["event_time"]),
        "diagnostic_only": kind == "placebo",
    }


def _tally(failures):
    out = {}
    for f in failures:
        key = f"{f.get('stage')}:{f.get('reason')}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))
