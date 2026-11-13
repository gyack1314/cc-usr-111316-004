"""面板装载与契约校验：平衡一户一年面板，G冻结、不可逆，折按district固定。

本模块只做形状与身份口径校验，不做任何拟合。数值以字符串保留，
经 Fraction 精确解析后在数值层按需转浮点。
"""
import json
from fractions import Fraction
from pathlib import Path

from .io import asset, integer

REQUIRED_COLUMNS = {"household", "district", "year", "G", "x", "outcome"}


class PanelError(ValueError):
    pass


def _fraction(value, field):
    if not isinstance(value, str):
        raise PanelError(f"{field}必须为字符串以保留精确数值，收到{type(value).__name__}")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise PanelError(f"{field}不是合法有理数: {value!r}")


def load_analysis(manifest, root=None):
    """读取清单与面板，返回经校验的 Analysis 数据对象。

    manifest: 已解析的清单 dict（或路径，路径时自动读取并校验资产身份）。
    root: 面板相对路径根；manifest 为路径时默认其父目录。
    """
    if isinstance(manifest, (str, Path)):
        from .io import load

        path = Path(manifest)
        manifest, _ = load(path)
        root = path.parent if root is None else root
    if root is None:
        raise PanelError("需要 root 以定位面板资产")
    cfg = validate_manifest(manifest)
    raw = asset(Path(root), manifest["panel"])
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != manifest["panel"]["rows"]:
        raise PanelError("面板行数与清单不符")
    return _build(rows, cfg)


def validate_manifest(v):
    """校验清单配置，返回规范化配置；不触碰面板行。"""
    if v.get("kind") != "staggered-att":
        raise PanelError("kind 必须为 staggered-att")
    L = integer(v["anticipation"], minimum=0)
    window = v.get("event_window", [0, 0])
    if not (isinstance(window, list) and len(window) == 2):
        raise PanelError("event_window 需为 [起, 止]")
    e_min, e_max = integer(window[0]), integer(window[1])
    if e_min > e_max:
        raise PanelError("event_window 起止颠倒")
    folds = v["folds"]
    if not isinstance(folds, dict) or not folds:
        raise PanelError("folds 必须为非空 district->折号 映射")
    fold_ids = set(folds.values())
    if any(type(x) is not int for x in fold_ids):
        raise PanelError("折号必须为整数")
    covariates = v["covariates"]
    if not isinstance(covariates, list) or not covariates or any(
        not isinstance(c, str) for c in covariates
    ):
        raise PanelError("covariates 必须为非空字符串列表")
    weights = v["population_weights"]
    if not isinstance(weights, dict) or not weights:
        raise PanelError("population_weights 缺失")
    for key, val in weights.items():
        _fraction(val, f"population_weights[{key}]")
        if Fraction(val) <= 0:
            raise PanelError(f"人口权重必须为正: {key}")
    bs = v["bootstrap"]
    integer(bs["replicates"], minimum=1)
    if bs.get("multiplier") not in ("Rademacher", "Mammen"):
        raise PanelError("multiplier 仅支持 Rademacher / Mammen")
    conf = bs.get("confidence", "19/20")
    level = _fraction(conf, "bootstrap.confidence")
    if not 0 < level < 1:
        raise PanelError("置信水平须在(0,1)")
    nuisance = v.get("nuisance", {})
    penalty = Fraction(_fraction(nuisance.get("logit_penalty", "0"), "logit_penalty"))
    if penalty != 0:
        raise PanelError("契约禁止未经登记的正则：logit_penalty 必须为 0")
    tol = Fraction(_fraction(nuisance.get("convergence_tolerance", "1/100000000"),
                             "convergence_tolerance"))
    return {
        "anticipation": L,
        "event_window": (e_min, e_max),
        "folds": dict(folds),
        "covariates": list(covariates),
        "weights_raw": dict(weights),
        "bootstrap_seed": integer(bs.get("seed", 0)),
        "bootstrap_replicates": integer(bs["replicates"], 1),
        "multiplier": bs["multiplier"],
        "confidence": float(level),
        "tolerance": float(tol),
    }


def _build(rows, cfg):
    if not rows:
        raise PanelError("面板为空")
    for r in rows:
        if not REQUIRED_COLUMNS <= set(r):
            raise PanelError(f"面板列缺失: {REQUIRED_COLUMNS - set(r)}")
    # 年份与索引
    years = sorted({r["year"] for r in rows})
    if any(type(y) is not int for y in years):
        raise PanelError("year 必须为整数")
    households = {}
    district_of = {}
    district_g = {}
    district_hh = {}
    seen = set()
    for r in rows:
        hh, dist, year, g = r["household"], r["district"], r["year"], r["G"]
        if not isinstance(hh, str) or not isinstance(dist, str):
            raise PanelError("household/district 必须为字符串")
        if (hh, year) in seen:
            raise PanelError(f"重复户-年: {hh} {year}")
        seen.add((hh, year))
        if g is not None and type(g) is not int:
            raise PanelError("G 必须为整数或 null(never)")
        if hh in district_of and district_of[hh] != dist:
            raise PanelError(f"家庭{hh}跨地区，聚类单位被破坏")
        district_of[hh] = dist
        if dist in district_g and district_g[dist] != g:
            raise PanelError(f"地区{dist}的G随时间变化，违反冻结/不可逆")
        district_g[dist] = g
        district_hh.setdefault(dist, set()).add(hh)
        households.setdefault(hh, {})[year] = r
    # 平衡：每户覆盖全部年份
    full_span = set(range(years[0], years[-1] + 1))
    if set(years) != full_span:
        raise PanelError("年份不连续")
    for hh, panel in households.items():
        if set(panel) != full_span:
            raise PanelError(f"家庭{hh}不是平衡面板")
    # 折与权重覆盖
    folds = cfg["folds"]
    missing_fold = set(district_g) - set(folds)
    if missing_fold:
        raise PanelError(f"地区缺少折分: {sorted(missing_fold)}")
    extra_fold = set(folds) - set(district_g)
    if extra_fold:
        raise PanelError(f"折分引用了不存在的地区: {sorted(extra_fold)}")
    weights = {}
    for hh in households:
        key = hh  # 清单键与 household 列一致
        if key not in cfg["weights_raw"]:
            raise PanelError(f"家庭缺少人口权重: {hh}")
        weights[hh] = Fraction(cfg["weights_raw"][key])
    # 解析协变量与结果，缓存 Fraction；x 可为标量字符串或多协变量字典
    cov_names = cfg["covariates"]
    parsed = {}
    for hh, panel in households.items():
        parsed[hh] = {}
        for year, r in panel.items():
            x = r["x"]
            if len(cov_names) == 1 and isinstance(x, str):
                xvals = (Fraction(x),)
            elif isinstance(x, dict):
                try:
                    xvals = tuple(Fraction(str(x[c])) for c in cov_names)
                except KeyError:
                    raise PanelError(f"{hh} {year} 协变量不完整")
            else:
                raise PanelError(f"{hh} {year} 的 x 形状与 covariates 不符")
            parsed[hh][year] = {
                "x": xvals,
                "outcome": Fraction(str(r["outcome"])),
            }
    cohorts = sorted(g for g in district_g.values() if g is not None)
    return Analysis(
        rows=rows,
        years=(years[0], years[-1]),
        households=households,
        district_of=district_of,
        district_g=district_g,
        district_hh=district_hh,
        folds={d: folds[d] for d in district_g},
        weights=weights,
        parsed=parsed,
        cohorts=tuple(sorted(set(cohorts))),
        never=tuple(sorted(d for d, g in district_g.items() if g is None)),
        cfg=cfg,
    )


class Analysis:
    """经校验的面板内存表示。"""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def L(self):
        return self.cfg["anticipation"]

    @property
    def year_min(self):
        return self.years[0]

    @property
    def year_max(self):
        return self.years[1]

    def base_year(self, g):
        """固定基期 b = g - L - 1。"""
        return g - self.L - 1

    def household_base_covariates(self, hh, b):
        return self.parsed[hh][b]["x"]

    def delta_outcome(self, hh, b, t):
        """变化结果 ΔY = Y_t - Y_b（精确分数）。"""
        return self.parsed[hh][t]["outcome"] - self.parsed[hh][b]["outcome"]

    def eligible_households(self, g, t, b):
        """只使用 b 与 t 两期都存在的家庭；平衡面板下即年份落在窗口内的家庭。"""
        if b < self.year_min or t > self.year_max or b >= t:
            return ()
        return tuple(hh for hh in self.households if b in self.parsed[hh] and t in self.parsed[hh])

    def legal_control_districts(self, t):
        """对照只能是 never 或 G > t + L；已处理与处于提前期的地区一律排除。"""
        L = self.L
        out = []
        for d, g in self.district_g.items():
            if g is None or g > t + L:
                out.append(d)
        return tuple(sorted(out))

    def treated_districts(self, g):
        return tuple(sorted(d for d, gg in self.district_g.items() if gg == g))

    def fold_of_district(self, d):
        return self.folds[d]
