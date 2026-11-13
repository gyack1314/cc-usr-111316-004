"""事件研究聚合与安慰剂诊断。

关键口径：
- 同一事件窗内，参与聚合的队列集合在所有事件时点保持不变；
  只纳入对窗内每个事件时点都可估的处理队列，避免随时点换队列制造趋势；
- 队列权重固定为该队列在固定基期 b=g-L-1 的处理家庭人口权重之和，
  不随事件时点变化；
- 实施前安慰剂只作诊断输出，不并入主事件研究，不以“通过检验”证明平行趋势。
"""


def _cohort_base_weight(an, g):
    b = an.base_year(g)
    return float(sum(
        an.weights[hh]
        for hh in an.households
        if an.district_g[an.district_of[hh]] == g
        and b in an.parsed[hh]
    ))


def build_post_grid(an):
    """主分析网格：每个队列 g 对 t>=g 且在面板内的格。"""
    grid = []
    for g in an.cohorts:
        for t in range(max(g, an.year_min), an.year_max + 1):
            if an.base_year(g) < an.year_min:
                continue
            grid.append((g, t))
    return grid


def build_placebo_grid(an, n_placebo=None):
    """安慰剂网格：处理前 t <= g-L-2 的格，基期取 t-1（诊断专用）。

    主分析的固定基期 b=g-L-1 在安慰剂时点之后，无法定义前期变化量；
    因此安慰剂是独立诊断：每个安慰剂格以 t-1 为基期，协变量同样取该基期，
    合法对照规则（never 或 G>t+L）不变。结果不并入主事件研究。
    返回 (g, t, base) 三元组。
    """
    L = an.L
    grid = []
    for g in an.cohorts:
        if an.base_year(g) < an.year_min:
            continue
        t_ceiling = g - L - 2
        candidates = [t for t in range(an.year_min + 1, t_ceiling + 1)]
        if n_placebo is not None:
            candidates = candidates[-n_placebo:]
        for t in candidates:
            grid.append((g, t, t - 1))
    return grid


def estimate_grid(an, grid, kind, trim=None, multipliers=None):
    """对网格逐格估计。post 网格为 (g,t)；placebo 网格为 (g,t,base)。"""
    from .cells import estimate_cell
    results = []
    for item in grid:
        if len(item) == 3:
            g, t, base = item
            results.append(estimate_cell(an, g, t, kind=kind, trim=trim,
                                          multipliers=multipliers, base_override=base))
        else:
            g, t = item
            results.append(estimate_cell(an, g, t, kind=kind, trim=trim,
                                          multipliers=multipliers))
    return results


def aggregate_event_study(cell_results, an, event_window=None, kind="post",
                          fixed_cohorts=None, fixed_weights=None):
    """把逐格 ATT 聚合为事件研究向量。

    event_window: (e_min, e_max)；主分析默认用清单窗口。
    参与队列 = 在窗口每个事件时点都有可估格的队列（固定集合）。
    fixed_cohorts/fixed_weights：bootstrap 注入原分析的固定队列与权重，
    保证重抽不改变聚合目标。
    """
    if event_window is None:
        event_window = an.cfg["event_window"]
    e_min, e_max = event_window
    by_event = {}
    not_estimable = []
    for r in cell_results:
        if r["kind"] != kind:
            continue
        e = r["event_time"]
        if not (e_min <= e <= e_max):
            continue
        if not r["estimable"]:
            not_estimable.append({
                "g": r["g"], "t": r["t"], "event_time": e,
                "reason": r["reason"], "stage": r["stage"]})
            continue
        by_event.setdefault(e, []).append(r)
    all_cohorts = {r["g"] for rows in by_event.values() for r in rows}
    event_points = list(range(e_min, e_max + 1))
    if fixed_cohorts is not None:
        participating = sorted(fixed_cohorts)
        excluded = {}
    else:
        participating = set(all_cohorts)
        excluded = {}
        for e in event_points:
            present = {r["g"] for r in by_event.get(e, [])}
            for g in all_cohorts - present:
                excluded.setdefault(g, []).append(e)
            participating &= present if present else set()
        participating = sorted(participating)
    if fixed_weights is not None:
        cohort_weights = dict(fixed_weights)
    else:
        cohort_weights = {g: _cohort_base_weight(an, g) for g in participating}
    w_sum = sum(cohort_weights.get(g, 0.0) for g in participating)
    rows_out = []
    fixed_mode = fixed_cohorts is not None
    for e in event_points:
        cells = [r for r in by_event.get(e, []) if r["g"] in set(participating)]
        if fixed_mode and {r["g"] for r in cells} != set(participating):
            missing = sorted(set(participating) - {r["g"] for r in cells})
            rows_out.append({
                "event_time": e, "estimable": False,
                "reason": "fixed_cohort_cell_not_estimable",
                "missing_cohorts": missing, "att": None})
            continue
        if not cells:
            rows_out.append({"event_time": e, "estimable": False,
                             "reason": "no_participating_cohort", "att": None})
            continue
        num = 0.0
        per_cohort = []
        for r in cells:
            cw = cohort_weights[r["g"]] / w_sum
            num += cw * r["att"]
            per_cohort.append({"g": r["g"], "att": r["att"],
                               "cohort_weight": cohort_weights[r["g"]],
                               "normalized_weight": cw,
                               "t": r["t"], "b": r["b"]})
        rows_out.append({
            "event_time": e, "estimable": True, "att": num,
            "per_cohort": per_cohort,
            "participating_cohorts": participating,
            "cohort_base_weights": cohort_weights,
        })
    return {
        "kind": kind,
        "event_window": [e_min, e_max],
        "participating_cohorts": participating,
        "cohort_base_weights": cohort_weights,
        "excluded_cohorts": [
            {"g": g, "missing_event_times": evs,
             "reason": "not_estimable_or_out_of_panel"}
            for g, evs in sorted(excluded.items())],
        "rows": rows_out,
        "not_estimable_cells": not_estimable,
        "diagnostic_only": kind == "placebo",
        "note": ("实施前安慰剂仅作诊断；不能以未拒绝零假设证明平行趋势。"
                 if kind == "placebo" else
                 "固定队列与固定基期人口权重，队列不随事件时点更换。"),
    }


def control_exit_schedule(an):
    """用户最关心的视图：每个 (g,t) 格哪些地区是合法对照、哪些在何时退出。"""
    rows = []
    all_districts = sorted(an.district_g)
    for g in an.cohorts:
        b = an.base_year(g)
        if b < an.year_min:
            continue
        for t in range(max(g, an.year_min), an.year_max + 1):
            legal = set(an.legal_control_districts(t))
            treated = set(an.treated_districts(g))
            detail = []
            for d in all_districts:
                gd = an.district_g[d]
                if d in treated:
                    status = "treated_cohort"
                elif d in legal:
                    status = "control"
                elif gd is not None and gd <= t + an.L:
                    if gd <= t:
                        status = "excluded_already_treated"
                    else:
                        status = "excluded_in_anticipation"
                else:
                    status = "excluded"
                detail.append({"district": d, "G": gd, "status": status})
            rows.append({
                "g": g, "t": t, "b": b, "event_time": t - g,
                "legal_control_districts": sorted(legal),
                "n_legal_control_districts": len(legal),
                "district_status": detail,
            })
    return rows
