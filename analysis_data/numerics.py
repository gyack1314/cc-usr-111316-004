"""纯标准库数值工具：精确秩检查、加权OLS、带分离检测的IRLS逻辑回归。

契约口径见 docs/data-contract.md：
- 不引入未经登记的正则；
- 真实完全分离或秩不足显式失败，不静默正则化；
- 协变量为小维精确值（字符串整数/分数），秩用 Fraction 精确消元判定，
  实际拟合用浮点，避免浮点容差把真正的秩不足判成满秩。
"""
from fractions import Fraction
from math import ceil, exp, log, sqrt


class FitFailure(Exception):
    """结构化拟合失败。code 取值：rank_deficient / separated / no_variance / not_converged。"""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def exact_rank(rows):
    """对矩阵（行向量列表）做 Fraction 高斯消元，返回精确秩。"""
    mat = [[Fraction(str(x)) for x in row] for row in rows]
    n_rows = len(mat)
    if n_rows == 0:
        return 0
    n_cols = len(mat[0])
    rank = 0
    row = 0
    for col in range(n_cols):
        pivot = None
        for r in range(row, n_rows):
            if mat[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        mat[row], mat[pivot] = mat[pivot], mat[row]
        pv = mat[row][col]
        mat[row] = [v / pv for v in mat[row]]
        for r in range(n_rows):
            if r != row and mat[r][col] != 0:
                factor = mat[r][col]
                mat[r] = [a - factor * b for a, b in zip(mat[r], mat[row])]
        row += 1
        rank += 1
        if row == n_rows:
            break
    return rank


def design_rank(X):
    """设计矩阵列秩（含截距时由调用方保证第一列恒为1）。"""
    if not X:
        return 0
    return exact_rank([[1.0] + list(row) for row in X])


def _solve(A, b):
    """高斯消元解 Ax=b；假定调用方已做精确秩检查，浮点奇异时仍显式失败。"""
    n = len(A)
    m = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-14:
            raise FitFailure("rank_deficient", f"浮点奇异于第{col}列")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        m[col] = [v / pv for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col]
                m[r] = [a - f * v for a, v in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def weighted_ols(X, y, w):
    """加权最小二乘，返回系数（首项为截距）。

    秩用精确消元判定；权重为正才参与。乘子bootstrap的零权重单位在调用前剔除，
    这里若仍出现非正权重视为口径错误。
    """
    n = len(X)
    if n == 0:
        raise FitFailure("rank_deficient", "空训练集")
    if any(x == 0 for x in w):
        raise FitFailure("rank_deficient", "OLS存在未剔除的零权重")
    Xd = [[1.0] + [float(v) for v in row] for row in X]
    k = len(Xd[0])
    if exact_rank(Xd) < k:
        raise FitFailure("rank_deficient", f"设计秩{exact_rank(Xd)}<参数数{k}")
    XtW = [[0.0] * n for _ in range(k)]
    for j in range(k):
        for i in range(n):
            XtW[j][i] = Xd[i][j] * w[i]
    A = [[sum(XtW[j][i] * Xd[i][q] for i in range(n)) for q in range(k)] for j in range(k)]
    bvec = [sum(XtW[j][i] * float(y[i]) for i in range(n)) for j in range(k)]
    return _solve(A, bvec)


def logistic_predict(beta, X):
    """返回概率向量，beta 首项为截距。数值截断到闭区间但不触碰0/1。"""
    out = []
    for row in X:
        z = beta[0] + sum(beta[j + 1] * float(v) for j, v in enumerate(row))
        if z >= 0:
            p = 1.0 / (1.0 + exp(-z))
        else:
            e = exp(z)
            p = e / (1.0 + e)
        out.append(min(1.0 - 1e-12, max(1e-12, p)))
    return out


def weighted_logit(X, y, w, tol=1e-8, max_iter=200):
    """IRLS 拟合带人口权重的逻辑回归，无惩罚。

    检测完全/准完全分离（系数发散、拟合概率贴边、迭代发散），
    以及单类、秩不足。返回 beta（首项截距）与收敛诊断。
    """
    n = len(X)
    if n == 0:
        raise FitFailure("separated", "空训练集")
    if any(x == 0 for x in w):
        raise FitFailure("separated", "logit存在未剔除的零权重")
    ones = sum(1 for v in y if v == 1)
    if ones == 0 or ones == n:
        raise FitFailure("separated", f"训练折只有单一类别(正例{ones}/{n})")
    Xd = [[1.0] + [float(v) for v in row] for row in X]
    k = len(Xd[0])
    if exact_rank(Xd) < k:
        raise FitFailure("rank_deficient", f"设计秩{exact_rank(Xd)}<参数数{k}")
    beta = [0.0] * k
    yf = [float(v) for v in y]
    prev_norm = 0.0
    diverging = 0
    for it in range(1, max_iter + 1):
        p = logistic_predict(beta, X)
        score = [0.0] * k
        H = [[0.0] * k for _ in range(k)]
        for i in range(n):
            wi = w[i] * p[i] * (1.0 - p[i])
            resid = w[i] * (yf[i] - p[i])
            for j in range(k):
                score[j] += Xd[i][j] * resid
                for q in range(k):
                    H[j][q] += Xd[i][j] * Xd[i][q] * wi
        try:
            delta = _solve(H, score)
        except FitFailure:
            raise FitFailure("separated", "IRLS信息矩阵奇异，疑似完全分离")
        new_beta = [beta[j] + delta[j] for j in range(k)]
        change = max(abs(d) for d in delta)
        edge = max(abs(logit(pv)) for pv in p)
        # 完全分离：系数范数持续发散且线性预测已极端，更新步长不缩小
        norm = sum(v * v for v in new_beta) ** 0.5
        if edge > 20.0 and norm > prev_norm and change > 1e-4:
            diverging += 1
            if diverging >= 5:
                raise FitFailure(
                    "separated",
                    f"系数持续发散(|beta|={norm:.2f}, max|logit p|={edge:.2f})，判定分离")
        else:
            diverging = 0
        prev_norm = norm
        beta = new_beta
        if change <= tol:
            p = logistic_predict(beta, X)
            edge = max(abs(logit(pv)) for pv in p)
            if edge > 35.0:
                raise FitFailure("separated", f"收敛于边界解(max|logit p|={edge:.2f})")
            return beta, {"iterations": it, "max_abs_linear_predictor": edge}
    raise FitFailure("not_converged", f"{max_iter}次迭代未达tol={tol}")


def logit(p):
    p = min(1.0 - 1e-12, max(1e-12, p))
    return log(p / (1.0 - p))


def mean(values, weights):
    sw = sum(weights)
    if sw <= 0:
        raise FitFailure("rank_deficient", "权重和非正")
    return sum(v * w for v, w in zip(values, weights)) / sw


def variance(values):
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(values) / n
    return sum((v - mu) ** 2 for v in values) / (n - 1)


def quantile_nearest_rank(values, level):
    """最近秩经验分位数（用于bootstrap同时带，确定性，无插值歧义）。"""
    if not values:
        raise FitFailure("not_converged", "空分位数输入")
    s = sorted(values)
    rank = max(1, min(len(s), ceil(level * len(s))))
    return s[rank - 1]


def safe_sqrt(v):
    return sqrt(v) if v > 0 else 0.0
