"""命令行：inspect 资料摘要 / estimate 分期ATT分析 / validate 合成核对。"""
import argparse
import json
from pathlib import Path

from .domain import summarize
from .estimator import run_analysis
from .io import load


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=_default))


def _default(o):
    from fractions import Fraction
    if isinstance(o, Fraction):
        return str(o)
    raise TypeError(f"不可序列化类型 {type(o)}")


def main():
    p = argparse.ArgumentParser(
        description="分期实施队列-时点 ATT 分析（纯标准库）")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("inspect", help="只展示资料身份与形状，不拟合")
    pi.add_argument("file")

    pe = sub.add_parser("estimate", help="格ATT、固定队列事件研究与同时带")
    pe.add_argument("file")
    pe.add_argument("--trim", type=float, default=None,
                    help="显式倾向修剪阈值(0,0.5)；不给则无支持即不可估")
    pe.add_argument("--no-bootstrap", action="store_true")
    pe.add_argument("--no-placebo", action="store_true")
    pe.add_argument("--n-placebo", type=int, default=None)
    pe.add_argument("--multiplier", choices=["Rademacher", "Mammen"],
                    default=None, help="聚类乘子；默认用清单设置")
    pe.add_argument("--out", type=Path, default=None, help="输出JSON路径")

    pv = sub.add_parser("validate", help="合成面板与扰动核对（恒定效应/缺支持/重排/缩窗）")
    pv.add_argument("--seed", type=int, default=20261113)
    pv.add_argument("--replicates", type=int, default=199)
    pv.add_argument("--out", type=Path, default=None)

    a = p.parse_args()
    if a.command == "inspect":
        v, h = load(a.file)
        _print({"manifest_sha256": h,
                "summary": summarize(v, Path(a.file).parent)})
    elif a.command == "estimate":
        package = run_analysis(
            a.file, trim=a.trim, n_placebo=a.n_placebo,
            do_bootstrap=not a.no_bootstrap,
            do_placebo=not a.no_placebo,
            multiplier=a.multiplier)
        text = json.dumps(package, ensure_ascii=False, indent=2, default=_default)
        if a.out:
            a.out.write_text(text, encoding="utf8")
            print(f"已写入 {a.out}")
        else:
            print(text)
    elif a.command == "validate":
        from .synthetic import run_validation_suite
        report = run_validation_suite(seed=a.seed, replicates=a.replicates)
        text = json.dumps(report, ensure_ascii=False, indent=2, default=_default)
        if a.out:
            a.out.write_text(text, encoding="utf8")
            print(f"已写入 {a.out}")
        else:
            print(text)


if __name__ == "__main__":
    main()
