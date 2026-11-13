"""分期实施 ATT 分析主编排：格估计 -> 固定队列事件研究 -> 聚类同时带。

输出是纯 dict / JSON 可序列化的“分析包”，逐格列出家庭、对照地区、基期、
折分、拟合诊断与聚合权重，并单列晚处理地区退出对照的时间表。
"""
from .aggregate import (
    aggregate_event_study,
    build_placebo_grid,
    build_post_grid,
    control_exit_schedule,
    estimate_grid,
)
from .inference import run_bootstrap
from .panel import load_analysis


def run_analysis(manifest, root=None, trim=None, n_placebo=None,
                 do_bootstrap=True, do_placebo=True, multiplier=None,
                 seed=None, replicates=None):
    """完整分析入口。

    manifest: 清单 dict 或清单路径。
    trim: None（默认，无支持即不可估）或 (0, 0.5) 内修剪阈值；
          修剪时单独报告新目标群体与剔除家庭。
    n_placebo: 安慰剂事件期数（取最靠近提前期的若干期）。
    multiplier/seed/replicates: 覆盖清单的 bootstrap 设置。
    """
    an = load_analysis(manifest, root)
    return _run_from_analysis(an, trim=trim, n_placebo=n_placebo,
                              do_bootstrap=do_bootstrap, do_placebo=do_placebo,
                              multiplier=multiplier, seed=seed,
                              replicates=replicates)


def _run_from_analysis(an, trim, n_placebo, do_bootstrap, do_placebo,
                       multiplier=None, seed=None, replicates=None):
    post_grid = build_post_grid(an)
    post_cells = estimate_grid(an, post_grid, "post", trim=trim)
    e_min, e_max = an.cfg["event_window"]
    post_event = aggregate_event_study(post_cells, an, (e_min, e_max), "post")
    # 固定聚合目标：bootstrap 全程沿用，不随乘子改变
    fixed_cohorts = post_event["participating_cohorts"]
    fixed_weights = post_event["cohort_base_weights"]

    package = {
        "config": {
            "anticipation_L": an.L,
            "base_year_rule": "b = g - L - 1",
            "event_window": [e_min, e_max],
            "covariates_measured_at_base": an.cfg["covariates"],
            "outcome_in_adjustment_set": False,
            "trim": trim,
            "estimand": ("ATT（修剪后新目标，见 trim_report）" if trim is not None
                         else "ATT：处理队列家庭，未改变目标群体"),
            "folds": an.folds,
        },
        "panel_overview": {
            "year_span": [an.year_min, an.year_max],
            "cohorts": list(an.cohorts),
            "never_treated_districts": list(an.never),
            "district_cohort": an.district_g,
            "n_districts": len(an.district_g),
            "n_households": len(an.households),
        },
        "cells": [_public_cell(r) for r in post_cells],
        "event_study": post_event,
        "control_exit_schedule": control_exit_schedule(an),
    }

    if do_bootstrap:
        def post_agg(cells, cohorts, weights):
            return aggregate_event_study(
                cells, an, (e_min, e_max), "post",
                fixed_cohorts=cohorts, fixed_weights=weights)

        package["bootstrap"] = run_bootstrap(
            an, post_grid, (e_min, e_max), post_agg, kind="post", trim=trim,
            fixed_cohorts=fixed_cohorts, fixed_weights=fixed_weights,
            multiplier=multiplier, seed=seed, replicates=replicates)

    if do_placebo:
        placebo_grid = build_placebo_grid(an, n_placebo)
        placebo_cells = estimate_grid(an, placebo_grid, "placebo", trim=trim)
        if placebo_grid:
            p_times = sorted(t - g for g, t, _ in placebo_grid)
            p_window = (min(p_times), max(p_times))
            p_event = aggregate_event_study(
                placebo_cells, an, p_window, "placebo")
            package["placebo"] = {
                "base_rule": "诊断专用：每格基期=t-1，独立于主分析固定基期",
                "cells": [_public_cell(r) for r in placebo_cells],
                "event_study": p_event,
            }
            if do_bootstrap:
                p_cohorts = p_event["participating_cohorts"]
                p_weights = p_event["cohort_base_weights"]

                def p_agg(cells, cohorts, weights):
                    return aggregate_event_study(
                        cells, an, p_window, "placebo",
                        fixed_cohorts=cohorts, fixed_weights=weights)

                package["placebo"]["bootstrap"] = run_bootstrap(
                    an, placebo_grid, p_window, p_agg, kind="placebo",
                    trim=trim, fixed_cohorts=p_cohorts, fixed_weights=p_weights,
                    multiplier=multiplier, seed=seed, replicates=replicates)
        else:
            package["placebo"] = {"cells": [], "note": "面板内无可构造的前期安慰剂格"}

    package["interpretation_guardrails"] = [
        "晚处理地区一旦满足 G<=t+L 即退出对照（先进入提前期、后成为已处理），"
        "绝不与早处理队列混作对照。",
        "事件研究在全事件窗使用同一批可估队列与固定基期人口权重；"
        "缩短窗口时共同事件期的数值应保持不变。",
        "实施前安慰剂仅作诊断，未拒绝零假设不等于平行趋势被证明。",
        "无支持域/秩不足/完全分离显式标记不可估，不静默裁权重；"
        "仅显式 trim 时改变目标群体并单独报告。",
        "bootstrap 有效次数不足或方差退化时不发布置信带，只给点估计与原因。",
    ]
    return package


def _public_cell(r):
    """格结果原样保留结构化字段，仅保证 JSON 友好（本项目数值已是原生类型）。"""
    return r
