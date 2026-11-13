# 分期适应项目的队列时点效应分析

本项目保存气候行动研究的合成资料与领域口径。`analysis_data.io`读取JSON及核验原始文件身份，`analysis_data.domain`核对基本形状，`analysis_data.cli`展示数量。详细量纲、时间、契约和统计口径见[资料说明](docs/data-contract.md)。地区、家庭、合同、价格及排放数据均为合成，不代表真实政策评价或投资建议。

## 分期实施 ATT 分析模块

针对“分三年落地、晚开工地区随后也受益”的分期实施设定，模块按**首次实施队列 g × 观测时点 t** 逐格估计，不再只报一个双向固定效应系数。核心口径：

- **格构造**：首次实施年 G 冻结、实施不可逆；统一提前反应期 L，固定基期 `b=g-L-1`，只用 b、t 两期都在的家庭；对照只能是 never 或 `G>t+L` 的地区，已处理与处于提前期的地区一律排除（见 `control_exit_schedule`，可直接查看晚处理地区何时退出对照）。
- **双重稳健 ATT**：因变量为变化结果 `ΔY=Y_t-Y_b`；对照家庭加权 OLS 拟合 `m0(X)`，全格 logit 给进入队列的倾向 `p(X)`，逆概率对照权重在**格内归一**；协变量只取基期时点，结果变量不进调整集。
- **地区交叉拟合**：折按 district 固定，任何一户及其同地区家庭不会同时进入训练与评估；无支持域、秩不足、完全分离显式返回 `estimable=false` 并带 `reason/stage`，不静默裁权重。仅当显式 `--trim` 时才修剪，并单独报告新目标群体与剔除家庭。
- **事件研究聚合**：同一事件窗使用同一批全程可估的处理队列与固定基期人口权重，队列不随事件时点更换；缩短窗口时共同事件期数值不变。
- **聚类乘子 bootstrap**：按地区生成乘子（默认 Rademacher 0/2，可选 Mammen），同地区全户同乘，重抽重做拟合并固定种子与折分；以 max-t 分位数构造同时置信带。处理地区过少、退化方差或有效重抽不足时**只给点估计与原因，不发布伪精确区间**。
- **实施前安慰剂仅作诊断**：每格以 `t-1` 为诊断基期，未拒绝零假设不等于平行趋势被证明。

纯标准库、无第三方依赖。项目根目录可运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q analysis_data tests
python -m analysis_data.cli inspect data/normal.json
python -m analysis_data.cli estimate data/normal.json
python -m analysis_data.cli estimate data/anticipation-removes-controls.json --multiplier Mammen
python -m analysis_data.cli validate
```

`estimate` 输出完整分析包：逐格家庭与对照地区、基期、折分、拟合诊断、归一权重、事件研究聚合、晚处理地区退出对照时间表、bootstrap 同时带与安慰剂诊断。`validate` 在已知恒定效应（真值+7）的合成面板上核对 ATT 恢复，并用缺支持/完全分离、地区标签重排、事件窗缩短、提前期退出对照、小面板拒发区间等场景验证结果与权重的含义。

`inspect`只展示资料，不做拟合、求解或分析发布。`invalid-config.json`包含基本配置错误，其余清单在身份上完整，可能包含需领域分析判断的不可识别、无界、冲突或不适用情况。有理数以字符串保留，null不等于0，随机种子属于分析设计。

Dockerfile固定Linux镜像；`docker build -t climate-analysis .`构建，`docker run --rm climate-analysis`显示摘要，`docker run --rm climate-analysis python -m unittest discover -s tests -v`运行已有读取检查。资料均可本地读取，运行不需要线上账户。
