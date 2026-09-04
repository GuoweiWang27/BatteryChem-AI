# BatteryChem-AI

## GreenBattery V3 · Reactivity and formulation compatibility

**Evidence-guided screening of polyphenol electrolyte additives for LFP‖Li cells.**

[Research question](#research-question) · [How V3 works](#how-v3-works) · [Run V3](#run-v3) · [Current limitations](#current-limitations) · [中文简介](#中文简介)

GreenBattery investigates whether reproducible improvements in battery cycling require two conditions to hold together: **electrochemical reactivity** and **compatibility with the electrolyte formulation**.

BatteryChem-AI is the computational part of this study. V3 organizes literature evidence, physical-affinity estimates and missing information into separate, testable predictions. Those predictions are then compared with laboratory measurements; they are not measurements of battery performance.

> **Development snapshot · 2026-09-03**
>
> The V3 implementation is on `codex/v3-prospective-prioritizer`. Experimental validation is pending. The current runtime is a transparent, deterministic rule-based system—not a trained model for predicting cycle life.

## Research question

The study concerns a fixed panel of five polyphenol additives in an ether-based electrolyte and LFP‖Li coin cells. The target formulation is 1 M LiFSI in DOL:DME at a 1:1 molar ratio, with a nominal additive loading of 5 wt%; actual preparation details belong to the experimental record.

The question is not simply which additive receives the highest score. It is:

> Within this fixed system, are reproducible cycling improvements confined to additives that satisfy both the reactivity and formulation-compatibility criteria?

The experimental comparison separates three observations:

- **Reactivity:** electrochemical behavior measured by cyclic voltammetry.
- **Compatibility:** solution appearance and conductivity relative to the additive-free baseline.
- **Cycling performance:** repeated-cell measurements, including capacity at cycle 100, retention and failures.

A match between prediction and experiment would support a particular claim within this system. A mismatch would prompt examination of evidence transfer, decision rules and experimental variability. Neither outcome establishes a universal rule for all electrolytes.

The five experimental candidates were fixed before V3 evaluation. They are **not** presented as a V3-selected top five.

## How V3 works

The [V3 source](src/batterychem_v3/) implements a literature-to-prediction workflow:

```text
Curated candidate records
    → reactivity and compatibility assessments
    → evidence uncertainty
    → eligible set Dpred
    → priority subset P, with at most three candidates
```

### Two separate assessments

| Component | What it evaluates | Implementation |
| --- | --- | --- |
| `Rpred` | Whether a candidate satisfies a registered, source-specific reactivity evidence pathway | [reactivity_head.py](src/batterychem_v3/reactivity_head.py) |
| `Cpred` | Target-electrolyte compatibility evidence, or a Hansen solubility parameter (HSP) physical-affinity estimate when its required inputs are available | [compatibility_head.py](src/batterychem_v3/compatibility_head.py) |
| `U` | Missing information, transfer between chemical contexts, reference-domain coverage and source conflict | [uncertainty.py](src/batterychem_v3/uncertainty.py) |

Reactivity evidence retains its source, reference electrode and medium. The current rules do not apply a universal numerical redox window to arbitrary molecules.

For the HSP pathway, the code calculates:

```text
Ra  = sqrt(4 × delta_d² + delta_p² + delta_h²)
RED = Ra / R0
```

Here, the three deltas are the supplied HSP differences. Without a valid `R0`, the code does not calculate RED or infer compatibility from distance alone. HSP estimates are proxies for physical affinity, not measurements of solubility, electrolyte stability or battery performance.

### Uncertainty and selection

The uncertainty score is a **rule-based evidence penalty**, not a calibrated probability or confidence interval:

```text
U = max(u_missing, u_transfer, u_ood, u_conflict)
```

In the current implementation, `U = 0` is labeled low, `U = 1` high, and intermediate values medium. Different missingness patterns can therefore produce the same overall score.

The eligible set `Dpred` requires:

1. `Rpred.class = pass`;
2. `Cpred.class = pass`, or `boundary-leaning` when both required boundary approvals are recorded;
3. uncertainty that is not classified as high.

Within `Dpred`, candidates are ordered using:

```text
joint_score = min(Rpred.score, Cpred.score) × (1 − U)
```

Ties follow the order specified in [priority_policy.json](config/v3_0/priority_policy.json). These scores encode rule classifications, not measured chemical quantities.

`P` contains at most three members of `Dpred`. It is not padded with ineligible candidates. An empty set is a valid output, but it does not establish that every candidate is experimentally ineffective.

### What the report contains

[report.py](src/batterychem_v3/report.py) produces JSON with:

- Per-candidate `Rpred`, `Cpred` and uncertainty components.
- Rule identifiers, evidence references and missing-information flags.
- Membership in `Dpred`, the priority subset `P`, and the selection budget.
- Configuration SHA-256 values and the state of the boundary-approval flags.

The report does **not** predict cycle-100 capacity, cycle life, SEI composition or an experimental improvement percentage.

## Three versions, three different questions

These are distinct research stages, not successive versions of one validated performance predictor.

| Stage | Main question | Scope |
| --- | --- | --- |
| **V1 — exploratory screening** | How do candidate formulations rank by predicted conductivity? | Historical XGBoost-based conductivity screening. It does not establish additive effects on cycling; historical output reproduction remains a separate task. |
| **V2 — base-electrolyte modeling** | How does conductivity vary with salt, solvent composition, concentration and temperature? | A separately archived CALiSol-23-based model, without additive-effect labels. That implementation is not included in this branch. |
| **V3 — current study** | How should reactivity, formulation compatibility and evidence uncertainty be evaluated together? | A fixed-panel, rule-based assessment intended for comparison with independent laboratory measurements. |

V3 does not use V1/V2 predictions or project cycling outcomes as scoring inputs. Earlier grid sizes, regression metrics and dashboard features are not V3 validation results.

## Current research status

| Item | Status |
| --- | --- |
| V3 rule engine, configuration and public synthetic example | Available in this branch |
| Five-candidate prediction files | Generated locally; individual predictions are not published here |
| Fingerprints of the confirmation document, experimental rules and prediction attachment | [Registered in Git](records/GB3-REG-20260902-01/) |
| Third-round prediction–experiment comparison | Pending; no validation results are reported here |
| Complete experimental analysis specification | Not yet complete; CV classification and the treatment of missing cycle-100 endpoints still require an explicit, dated specification |

The fingerprint registration was pushed to GitHub on **2026-09-02 at 23:56:06 UTC+08:00** ([commit `ace5148`](https://github.com/GuoweiWang27/BatteryChem-AI/commit/ace5148632d6bbf397b9e43680d4a77944474fdf)). It publishes filenames and SHA-256 values, not the underlying PDF or ZIP contents. The confirmation document was unsigned at registration.

This record supports file-version chronology. It is not an experimental result, a witness signature or, by itself, proof of outcome blinding.

## Run V3

The V3 CLI uses the Python standard library. The commands below target Python 3.12 and the V3 branch; the legacy machine-learning dependencies are not needed for this example.

### 1. Get the V3 branch

```bash
git clone --branch codex/v3-prospective-prioritizer \
  https://github.com/GuoweiWang27/BatteryChem-AI.git
cd BatteryChem-AI
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with `.venv\Scripts\activate` and set `PYTHONPATH` to `src` using your shell's syntax.

### 2. Run the public synthetic example

The following is a macOS/Linux shell example. It writes the output to a new temporary directory.

```bash
batterychem_demo_dir="$(mktemp -d)"
PYTHONPATH=src python3 -m batterychem_v3.cli \
  --input tests/v3/synthetic_cli_input.json \
  --config-dir config/v3_0 \
  --mode RULE \
  --output "$batterychem_demo_dir/report.json"
python3 -m json.tool "$batterychem_demo_dir/report.json"
```

The supplied record is **fictional test input**, not a molecule or an experimental result. A shortened view of its output is:

```json
{
  "schema_version": "batterychem_v3_report_v1",
  "mode": "RULE",
  "algorithm": "RULE_TRANSPARENT_V3",
  "Dpred": ["SYN-DET-001"],
  "P": ["SYN-DET-001"],
  "budget_k": 3
}
```

This example exercises software behavior only. Its reactivity window uses a synthetic unit and must not be reused as a chemical threshold.

### 3. Run the tests

```bash
python3 -m pip install pytest==7.4.4
python3 -m pytest -q tests/v3/test_contract.py
```

The suite covers input checks, missing-value handling, rule paths, deterministic serialization, priority-set behavior and output boundaries. Passing software tests is not experimental validation or uncertainty calibration.

### Inputs and configuration

- Start with the [synthetic input structure](tests/v3/synthetic_cli_input.json) and [feature registry](config/v3_0/feature_registry.csv).
- Real inputs require independently checked chemical identities, sources, units and context. The CLI neither retrieves papers nor computes molecular descriptors from SMILES.
- Use the bundled `config/v3_0` directory. The HSP implementation reads its RED bands from the repository's threshold registry; substituting another `--config-dir` does not replace every rule source.
- The boundary flags represent previously obtained approvals. They are ordinary command-line booleans, not digital signatures or a way to create an approval.

The public checkout can reproduce the supplied synthetic example. It does not contain the complete private input-and-evidence packet needed to reproduce the study's real-candidate predictions.

## Current limitations

- **Fixed-panel implementation.** Some reactivity pathways are candidate- and source-specific, and input validation currently requires `wet_lab_panel=true` and `selected_by_v3=false`. General virtual-library screening is not supported by this interface.
- **Uncalibrated uncertainty.** Missing domain definitions or transfer-context assignments can dominate `U`. A high score is an evidence warning, not a measured probability of chemical failure.
- **Incomplete input safeguards.** Unknown source-conflict values currently contribute zero to the conflict component, and some declared required fields and enum restrictions are not fully enforced. Curated inputs still need explicit checking.
- **No learned V3 backend.** `LOGISTIC_AUGMENTED` has an external-data eligibility check, but no logistic model is fitted or applied; report generation still uses the rule algorithm. Use `RULE` for the documented workflow.
- **Configuration status is not a signature.** Some configuration metadata retains draft labels. A repository checkout does not establish human approval or complete preregistration; those belong to the identified study records.

## Repository map

```text
src/batterychem_v3/        V3 input checks, two-axis rules, uncertainty and CLI
config/v3_0/              Feature/threshold registries and decision policies
tests/v3/                 Synthetic fixture and software tests
records/                  Published material fingerprints
reports/                  Historical engineering records
app.py                    Legacy conductivity-model entry point
search_all_families.py     Legacy formulation-grid screening
web_app.py                 Legacy web interface, not a V3 interface
requirements.txt          Legacy ML/UI stack and test dependencies
```

The root-level legacy scripts remain for historical continuity. Use `python3 -m batterychem_v3.cli` for V3; `app.py` and the old web interface do not invoke the V3 engine.

## Next steps

1. Complete the experimental measurement and analysis specifications, recording when each decision was made relative to access to the corresponding outcomes.
2. Compare the registered predictions with the laboratory observations, reporting both matches and counterexamples and retaining failed cells.
3. Address documented implementation limitations in a separately versioned revision, without overwriting the registered predictions.
4. Consider a learned extension only when compatible external labels and an appropriate validation design are available.

## 中文简介

### 现在研究的是什么？

GreenBattery 研究的是醚系 LFP‖Li 电池中的多酚添加剂：在固定配方条件下，可重复的循环改善，是否需要同时满足“电化学反应性”和“配方相容性”两类条件？

BatteryChem-AI 是其中的计算辅助部分。V3 把文献证据、物理相容性估计和缺失信息分开记录，形成可以与实验逐项比较的预测。它不是直接预测电池寿命的机器学习模型，也不把计算分数当作实验结果。

### V1、V2、V3 有什么区别？

- **V1：**历史电导率筛选与候选生成探索，不能直接说明循环性能。
- **V2：**基于 CALiSol-23 的基础电解液电导率建模，不含添加剂效应标签；其代码另行归档。
- **V3：**对固定候选组分别评估反应性、相容性和证据不确定性，再形成可行集合 `Dpred` 与最多三项的优先集合 `P`。

这五种实验候选不是 V3 从大库中选出的 Top 5。V3 可以输出空集，但空集不等于所有候选在实验中都无效；不确定性分数也不是校准后的成功概率。

### 当前进度与使用边界

V3 代码与合成示例已公开；真实候选预测保留在本地，第三轮实验对照尚未在本仓库报告。GitHub 目前只登记确认书、判定规则和预测附件包的文件名与校验值，未公开原文件。确认书登记时为待签版本，CV 分类与缺失终点统计仍有待明确的事项。

运行方法见上方 [Run V3](#run-v3)。示例只验证程序行为；候选专属规则、未经校准的不确定性及尚未完善的输入检查，见 [Current limitations](#current-limitations)。原有 `app.py`、网页界面和依赖列表属于早期实现，不代表 V3 已有对应功能。

## License and citation

The repository's software is distributed under the [MIT License](LICENSE). Third-party datasets and publications retain their own terms.

To reference the software, cite **Guowei Wang, BatteryChem-AI / GreenBattery V3 (2026)** and the exact commit used. No published-paper citation or DOI is claimed for the current V3 study.
