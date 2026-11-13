"""逐格 (g,t) 双重稳健 ATT：地区折交叉拟合、格内归一逆概率对照权重。

口径（docs/data-contract.md）：
- D=1(G=g)；合法对照 = never 或 G>t+L，已处理/提前期地区不得混入；
- 固定基期 b=g-L-1，只用 b、t 两期都在的家庭；协变量取 b 时点，
  结果变量只以变化量 ΔY=Y_t-Y_b 进入，不进调整集；
- 对照 OLS 拟合 m0(X)=E[ΔY|D=0,X]，全样本 logit 给倾向 p(X)；
- 折按 district 固定，训练折不得包含评估折的任何地区；
- 无支持域、秩不足、分离显式返回不可估，绝不静默裁权重改目标群体；
  仅当调用方显式给 trim 时才修剪，并单独报告新目标与剔除家庭。
"""
from fractions import Fraction

from .numerics import (
    FitFailure,
    design_rank,
    logistic_predict,
    weighted_logit,
    weighted_ols,
)

PROP_BOUND = 1e-6  # 倾向贴边阈值：越过即判定无支持，不参与归一


def _not_estimable(g, t, b, kind, reason, stage, **extra):
    out = {
        "g": g, "t": t, "b": b, "event_time": t - g, "kind": kind,
        "estimable": False, "reason": reason, "stage": stage, "att": None,
    }
    out.update(extra)
    return out


def _empirical_support(treated_x, control_x):
    """离散低基数协变量的经验公共支持：处理组每个模式至少在对照出现一次。

    高基数/连续协变量下精确模式重合是过强要求，返回跳过标记，
    改由折外倾向得分的贴边检测（PROP_BOUND）把关支持域。
    """
    all_rows = treated_x + control_x
    distinct = {tuple(Fraction(v) for v in row) for row in all_rows}
    n_units = len(all_rows)
    # 模式数接近单位数即视为高基数，精确模式支持不适用
    if len(distinct) > max(8, int(0.5 * n_units)):
        return True, []
    control_patterns = set(tuple(Fraction(v) for v in row) for row in control_x)
    missing = sorted(
        str(tuple(float(v) for v in row))
        for row in {tuple(r) for r in treated_x}
        if tuple(Fraction(v) for v in row) not in control_patterns
    )
    return (not missing), missing


def _cell_units(an, g, t, b, kind):
    """构造格内单位；返回 (units, meta) 或不可估原因字符串。"""
    if b < an.year_min:
        return None, f"基期b={b}早于面板起点{an.year_min}"
    if t > an.year_max:
        return None, f"时点t={t}晚于面板终点{an.year_max}"
    if b >= t:
        return None, f"基期b={b}不早于t={t}（变化结果未定义）"
    treated_districts = an.treated_districts(g)
    if not treated_districts:
        return None, f"队列g={g}没有处理地区"
    control_districts = an.legal_control_districts(t)
    if not control_districts:
        return None, f"t={t}不存在合法对照地区（never或G>{t + an.L}）"
    eligible = set(an.eligible_households(g, t, b))
    treated_hh = sorted(
        hh for hh in eligible if an.district_g[an.district_of[hh]] == g
    )
    control_hh = sorted(
        hh for hh in eligible if an.district_of[hh] in set(control_districts)
    )
    if not treated_hh or not control_hh:
        return None, "格内处理或对照家庭为空"
    units = []
    for hh in treated_hh:
        units.append(_make_unit(an, hh, b, t, 1))
    for hh in control_hh:
        units.append(_make_unit(an, hh, b, t, 0))
    support_ok, missing_patterns = _empirical_support(
        [u["x"] for u in units if u["d"] == 1],
        [u["x"] for u in units if u["d"] == 0],
    )
    meta = {
        "treated_districts": list(treated_districts),
        "control_districts": list(control_districts),
        "treated_households": treated_hh,
        "control_households": control_hh,
        "missing_support_patterns": missing_patterns,
        "support_ok": support_ok,
    }
    if not support_ok:
        return None, meta
    return units, meta


def _make_unit(an, hh, b, t, d):
    return {
        "household": hh,
        "district": an.district_of[hh],
        "fold": an.fold_of_district(an.district_of[hh]),
        "d": d,
        "x": tuple(float(v) for v in an.household_base_covariates(hh, b)),
        "dy": float(an.delta_outcome(hh, b, t)),
        "w": float(an.weights[hh]),
    }


def _apply_trim(units, trim):
    """显式修剪：保留 trim <= p <= 1-trim 的家庭；返回(保留, 剔除报告)。"""
    if trim is None:
        return units, None
    if not 0 < trim < 0.5:
        raise ValueError("trim 必须在 (0, 0.5) 内")
    # 修剪需要先有倾向；用全样本 logit 决定修剪集合，再在修剪样本上交叉拟合
    kept, excluded = [], []
    try:
        beta, diag = weighted_logit(
            [u["x"] for u in units], [u["d"] for u in units],
            [u["w"] for u in units],
        )
        ps = logistic_predict(beta, [u["x"] for u in units])
    except FitFailure as f:
        return None, {"stage": "trim_screen_logit", "reason": f.code, "detail": f.detail}
    for u, p in zip(units, ps):
        if trim <= p <= 1 - trim:
            kept.append(u)
        else:
            excluded.append({
                "household": u["household"], "district": u["district"],
                "d": u["d"], "p": p, "rule": f"p outside [{trim}, {1 - trim}]",
            })
    report = {
        "stage": None,
        "trim": trim,
        "excluded_households": excluded,
        "excluded_districts": sorted({e["district"] for e in excluded}),
        "new_target_treated_households": sorted(
            u["household"] for u in kept if u["d"] == 1),
        "new_target_treated_districts": sorted(
            {u["district"] for u in kept if u["d"] == 1}),
        "screen_logit_iterations": diag["iterations"],
    }
    if not any(u["d"] == 1 for u in kept) or not any(u["d"] == 0 for u in kept):
        report["stage"] = "trim"
        report["reason"] = "trim_after_empty_class"
        return None, report
    return kept, report


def estimate_cell(an, g, t, kind="post", trim=None, multipliers=None,
                  base_override=None):
    """估计单个 (g,t) 格。multipliers: 可选 district->非负乘子（bootstrap）。

    base_override: 安慰剂诊断专用基期；主分析恒为 b=g-L-1。
    返回结构化 dict；不可估时 estimable=False 且带 reason/stage，
    不抛异常、不静默改样本。
    """
    L = an.L
    b = an.base_year(g) if base_override is None else base_override
    if kind == "placebo":
        if g - t <= L:
            return _not_estimable(g, t, b, kind,
                                  f"安慰剂t={t}落入提前期(g-t={g - t}<=L={L})",
                                  "placebo_window")
        if b != t - 1:
            return _not_estimable(g, t, b, kind,
                                  f"安慰剂基期须为t-1={t - 1}，收到{b}",
                                  "placebo_window")
    units, meta = _cell_units(an, g, t, b, kind)
    if units is None:
        reason = meta if isinstance(meta, str) else "no_common_support"
        extra = meta if isinstance(meta, dict) else {}
        return _not_estimable(g, t, b, kind, reason, "cell_construction", **extra)
    trim_report = None
    if trim is not None:
        units, trim_report = _apply_trim(units, trim)
        if units is None:
            return _not_estimable(g, t, b, kind,
                                  trim_report.get("reason", "trim_failure"),
                                  "trim", trim_report=trim_report,
                                  **{k: v for k, v in meta.items()})
    if multipliers is not None:
        units = [dict(u, w=u["w"] * float(multipliers[u["district"]])) for u in units]
        if all(u["w"] == 0 for u in units):
            return _not_estimable(g, t, b, kind, "乘子使格内权重全为0",
                                  "bootstrap_multiplier",
                                  **meta)
    fold_ids = sorted({u["fold"] for u in units})
    treated_folds = {u["fold"] for u in units if u["d"] == 1}
    control_folds = {u["fold"] for u in units if u["d"] == 0}
    if len(treated_folds) < 2 or len(control_folds) < 2:
        return _not_estimable(
            g, t, b, kind,
            f"交叉拟合需要处理、对照各至少2个折，实得{len(treated_folds)}/{len(control_folds)}",
            "fold_support", fold_ids=fold_ids, **meta)

    # 折外预测
    oof = {u["household"]: None for u in units}
    fold_diag = []
    for k in fold_ids:
        train = [u for u in units if u["fold"] != k]
        # 乘子精确为0的单位不贡献矩条件/似然，训练与评估均剔除；
        # 负乘子（Mammen）是合法权重必须保留；剔除后缺类别/秩不足仍显式失败。
        train_pos = [u for u in train if u["w"] != 0]
        evalf = [u for u in units if u["fold"] == k and u["w"] != 0]
        tr_t = [u for u in train_pos if u["d"] == 1]
        tr_c = [u for u in train_pos if u["d"] == 0]
        diag = {"fold": k,
                "n_train_treated": len(tr_t), "n_train_control": len(tr_c),
                "n_eval_treated": sum(1 for u in evalf if u["d"] == 1),
                "n_eval_control": sum(1 for u in evalf if u["d"] == 0),
                "n_zero_weight_dropped": {
                    "train": len(train) - len(train_pos),
                    "eval": sum(1 for u in units if u["fold"] == k) - len(evalf)}}
        if not tr_t or not tr_c:
            return _not_estimable(g, t, b, kind,
                                  f"折{k}训练集缺类别(处理{len(tr_t)}/对照{len(tr_c)})",
                                  "fold_support", fold_diagnostics=fold_diag + [diag],
                                  **meta)
        if not evalf:
            return _not_estimable(g, t, b, kind,
                                  f"折{k}评估单位被乘子全部置零",
                                  "fold_support", fold_diagnostics=fold_diag + [diag],
                                  **meta)
        Xtr = [u["x"] for u in train_pos]
        diag["design_rank"] = design_rank(Xtr)
        diag["n_parameters"] = len(Xtr[0]) + 1
        # m0：仅对照家庭的加权OLS
        try:
            beta_ols = weighted_ols(
                [u["x"] for u in tr_c], [u["dy"] for u in tr_c],
                [u["w"] for u in tr_c])
        except FitFailure as f:
            diag["ols_failure"] = {"code": f.code, "detail": f.detail}
            return _not_estimable(g, t, b, kind, f.detail, f"ols_fold{k}",
                                  fold_diagnostics=fold_diag + [diag], **meta)
        # p：全训练集 logit
        try:
            beta_logit, ldiag = weighted_logit(
                Xtr, [u["d"] for u in train_pos], [u["w"] for u in train_pos],
                tol=an.cfg["tolerance"])
        except FitFailure as f:
            diag["logit_failure"] = {"code": f.code, "detail": f.detail}
            return _not_estimable(g, t, b, kind, f.detail, f"logit_fold{k}",
                                  fold_diagnostics=fold_diag + [diag], **meta)
        diag["logit_iterations"] = ldiag["iterations"]
        diag["logit_max_abs_linear_predictor"] = ldiag["max_abs_linear_predictor"]
        p_eval = logistic_predict(beta_logit, [u["x"] for u in evalf])
        m0_eval = [beta_ols[0] + sum(beta_ols[j + 1] * v for j, v in enumerate(u["x"]))
                   for u in evalf]
        for u, p, m0 in zip(evalf, p_eval, m0_eval):
            oof[u["household"]] = {"p": p, "m0": m0}
        fold_diag.append(diag)

    # 仍有权重为正但未获得折外预测的单位（如整折评估被置零），显式失败
    uncovered = [u["household"] for u in units
                 if u["w"] != 0 and oof[u["household"]] is None]
    if uncovered:
        return _not_estimable(g, t, b, kind,
                              f"折{uncovered and an.fold_of_district(an.district_of[uncovered[0]])}"
                              f"评估单位缺失折外预测: {uncovered[:4]}",
                              "fold_support", fold_diagnostics=fold_diag, **meta)

    # 支持域：折外倾向不得贴边（零乘子单位不参与，跳过）
    for u in units:
        if u["w"] == 0:
            continue
        s = oof[u["household"]]
        if s is None:
            return _not_estimable(g, t, b, kind,
                                  f"家庭{u['household']}缺失折外预测（折评估可能被置零）",
                                  "fold_support",
                                  fold_diagnostics=fold_diag, **meta)
        if u["d"] == 1 and s["p"] >= 1 - PROP_BOUND:
            return _not_estimable(g, t, b, kind,
                                  f"处理家庭{u['household']}折外p={s['p']:.3e}贴1，无支持",
                                  "no_common_support", fold_diagnostics=fold_diag, **meta)
        if u["d"] == 0 and s["p"] <= PROP_BOUND:
            return _not_estimable(g, t, b, kind,
                                  f"对照家庭{u['household']}折外p={s['p']:.3e}贴0，无支持",
                                  "no_common_support", fold_diagnostics=fold_diag, **meta)

    # 格内归一：处理组按人口权重；对照按 w*p/(1-p)。零乘子单位不参与，
    # 负乘子（Mammen）保留，因此权重和为零/负时该重抽不可估。
    active = [u for u in units if u["w"] != 0]
    treated = [u for u in active if u["d"] == 1]
    control = [u for u in active if u["d"] == 0]
    sw_t = sum(u["w"] for u in treated)
    raw_c = [(u, u["w"] * oof[u["household"]]["p"] /
              (1 - oof[u["household"]]["p"])) for u in control]
    sw_c = sum(r for _, r in raw_c)
    if sw_t == 0 or sw_c == 0:
        return _not_estimable(g, t, b, kind, "归一权重和非正", "weights",
                              fold_diagnostics=fold_diag, **meta)
    att_t = sum((u["w"] / sw_t) *
                (u["dy"] - oof[u["household"]]["m0"]) for u in treated)
    att_c = sum((r / sw_c) *
                (u["dy"] - oof[u["household"]]["m0"]) for u, r in raw_c)
    att = att_t - att_c

    # 权重诊断
    ipw_norm = [(u["household"], r / sw_c) for u, r in raw_c]
    tw_norm = [(u["household"], u["w"] / sw_t) for u in treated]
    ess_c = 1 / sum(q ** 2 for _, q in ipw_norm)
    weight_diag = {
        "treated_pop_weight_sum": sw_t,
        "control_ipw_raw_sum": sw_c,
        "treated_normalized_weights": [
            {"household": h, "weight": q} for h, q in tw_norm],
        "control_normalized_ipw": [
            {"household": h, "weight": q} for h, q in
            sorted(ipw_norm, key=lambda z: -z[1])],
        "max_control_ipw": max(q for _, q in ipw_norm),
        "control_ess": ess_c,
        "control_n": len(control),
        "min_oof_p": min(oof[u["household"]]["p"] for u in active),
        "max_oof_p": max(oof[u["household"]]["p"] for u in active),
    }
    return {
        "g": g, "t": t, "b": b, "event_time": t - g, "kind": kind,
        "estimable": True, "reason": None, "stage": None,
        "att": att,
        "att_treated_component": att_t,
        "att_control_component": att_c,
        "n_treated_households": len(treated),
        "n_control_households": len(control),
        "treated_districts": meta["treated_districts"],
        "control_districts": meta["control_districts"],
        "treated_households": meta["treated_households"],
        "control_households": meta["control_households"],
        "fold_ids": fold_ids,
        "fold_diagnostics": fold_diag,
        "weight_diagnostics": weight_diag,
        "trim_report": trim_report,
    }
