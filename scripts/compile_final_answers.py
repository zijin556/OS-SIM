from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io


def load_json(rel_path: str) -> Any:
    with (ROOT / rel_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_num(value: Any, ndigits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return str(value)


def peak_text(peak: dict[str, Any] | None) -> str:
    if not peak:
        return "n/a"
    return (
        f"(fy={fmt_num(peak.get('freq_y_cycles_per_px'))}, "
        f"fx={fmt_num(peak.get('freq_x_cycles_per_px'))})"
    )


def best_forward_model(forward: dict[str, Any], group: str) -> tuple[str, dict[str, Any]]:
    losses = forward["models"][group]["losses"]
    model_name = min(losses, key=lambda name: losses[name].get("mod_rmse", float("inf")))
    return model_name, losses[model_name]


def build_answers() -> dict[str, Any]:
    manifest = load_json("outputs/data_manifest.json")
    baseline = load_json("outputs/baseline/t6_hv/metrics.json")
    frequency = load_json("outputs/frequency/t6_hv/metrics.json")
    group_compare = load_json("outputs/frequency/group_compare/t6_vs_t12_compare.json")
    forward = load_json("outputs/forward/t6_hv/model_compare.json")
    ablation = load_json("outputs/ablation/t6_hv/metrics.json")
    platform_compare = load_json("outputs/height/platform_compare.json")
    audit = load_json("outputs/audit/coverage.json")
    z_unit = load_json("outputs/z_unit/evidence.json")
    height_truth_assets = load_json("outputs/height_truth_assets/assets.json")

    fine = next(s for s in manifest["samples"] if s["name"] == "细台阶")
    excluded = manifest.get("excluded_samples", [])
    complete = [
        s["name"]
        for s in manifest["samples"]
        if not s.get("excluded") and s.get("incomplete_layer_count") == 0
    ]
    h_freq = frequency["groups_detail"]["ossim_t6_h_3step"]
    v_freq = frequency["groups_detail"]["ossim_t6_v_3step"]
    h_best, h_best_metrics = best_forward_model(forward, "ossim_t6_h_3step")
    v_best, v_best_metrics = best_forward_model(forward, "ossim_t6_v_3step")
    h_level0 = forward["models"]["ossim_t6_h_3step"]["losses"]["level0_ideal_cos_group_period"]
    v_level0 = forward["models"]["ossim_t6_v_3step"]["losses"]["level0_ideal_cos_group_period"]
    h_measured = forward["models"]["ossim_t6_h_3step"]["losses"]["level2_measured_dmd_calibrated_map"]
    v_measured = forward["models"]["ossim_t6_v_3step"]["losses"]["level2_measured_dmd_calibrated_map"]
    baseline_row = next(row for row in ablation["a_stack_ablation"] if row["ablation"] == "A_baseline_A_cls_only")
    aggressive_row = next(row for row in ablation["a_stack_ablation"] if row["ablation"] == "P5_aggressive_A_only_grid_loss")
    measured_platform_row = next(
        row for row in ablation["a_stack_ablation"] if row["ablation"] == "P5_measured_calibrated_platform_constrained"
    )
    generated_platform_row = next(
        row for row in ablation["a_stack_ablation"] if row["ablation"] == "P5_platform_constrained_early_stop"
    )
    method_by_name = {row["method"]: row for row in platform_compare["methods"]}
    measured_platform_method = method_by_name["measured_calibrated_platform_constrained"]
    height_truth_candidate_count = height_truth_assets.get("candidate_count", 0)
    height_truth_pseudo_count = height_truth_assets.get("pseudo_reference_candidate_count", 0)
    height_truth_found = bool(height_truth_assets.get("independent_truth_found"))

    answers = [
        {
            "id": 1,
            "question": "数据维度是否解析正确？",
            "answer": (
                "是。`细台阶` 有 31 个完整 z 层；T6 H/V 分别使用帧 100-102 和 103-105，"
                "组合为 `Y_real[k,g,m,y,x]`，主实验 crop 为 `31 x 2 x 3 x 256 x 256`。"
                f"`台阶曝光1` 保持排除；完整对比样本为 {', '.join(complete)}。"
            ),
            "evidence": ["outputs/data_manifest.json", "outputs/data_dimension_inference.json"],
            "status": "answered",
        },
        {
            "id": 2,
            "question": "传统 OS-SIM baseline 结果如何？",
            "answer": (
                "baseline 已输出 H/V 单方向和 fused 光切片。`A_fused_stack` 为 "
                f"[{baseline['z_count']}, {baseline['shape'][0]}, {baseline['shape'][1]}]，fused peakfit 高度 std 为 "
                f"{fmt_num(baseline['fused']['height_std'], 4)} um，平台台阶估计为 "
                f"{fmt_num(baseline_row['step_height_scan_units'], 5)} um，平台 RMS 为 "
                f"{fmt_num(baseline_row['platform_rms'], 6)}。"
            ),
            "evidence": ["outputs/baseline/t6_hv/metrics.json", "outputs/height/platform_compare.json"],
            "status": "answered",
        },
        {
            "id": 3,
            "question": "实拍 A_cls 中的残余网格/条纹频率是什么？",
            "answer": (
                "H 主条纹峰为 "
                f"{peak_text(h_freq['selected_nominal_main_peak'])}，V 主条纹峰为 "
                f"{peak_text(v_freq['selected_nominal_main_peak'])}。A_cls 的主要残余网格/Moire 峰集中在 "
                f"H {peak_text(h_freq['section_residual_peaks'][0])} 和 "
                f"V {peak_text(v_freq['section_residual_peaks'][0])}，grid energy 均值分别为 "
                f"{fmt_num(h_freq['grid_energy_mean'])} 和 {fmt_num(v_freq['grid_energy_mean'])}。"
            ),
            "evidence": ["outputs/frequency/t6_hv/metrics.json", "outputs/frequency/t6_hv/frequency_report.md"],
            "status": "answered",
        },
        {
            "id": 4,
            "question": "T=6 三步和 T=12 哪个更好，证据是什么？",
            "answer": (
                "当前 256 crop 的诊断排序选择 `t6_3step`。综合 grid/sharpness score 中 "
                f"`t6_3step={fmt_num(group_compare['ranking']['score']['t6_3step'])}`，"
                f"`t12_6step={fmt_num(group_compare['ranking']['score']['t12_6step'])}`，"
                f"`t12_3step={fmt_num(group_compare['ranking']['score']['t12_3step'])}`。"
                "该结论只基于频域残余和峰锐度，不是外部高度真值验证。"
            ),
            "evidence": ["outputs/frequency/group_compare/t6_vs_t12_compare.json"],
            "status": "answered",
        },
        {
            "id": 5,
            "question": "理想 cos 模型能否重投影真实图像？",
            "answer": (
                "不能充分解释真实 crop。Level-0 ideal cos 的 modulation RMSE 为 "
                f"H {fmt_num(h_level0['mod_rmse'])}、V {fmt_num(v_level0['mod_rmse'])}，"
                "且残差 FFT 仍有结构性峰；它只能作为低阶对照。"
            ),
            "evidence": ["outputs/forward/t6_hv/model_compare.json", "outputs/forward/t6_hv/forward_report.md"],
            "status": "answered",
        },
        {
            "id": 6,
            "question": "加入 DMD-CCD 映射后重投影误差是否下降？",
            "answer": (
                "方向相关。H 方向明显下降：ideal/generated 约 0.1468，measured calibrated 为 "
                f"{fmt_num(h_measured['mod_rmse'])}，最佳带 blur 为 {fmt_num(h_best_metrics['mod_rmse'])} "
                f"({h_best})。V 方向 measured calibrated RMSE 为 {fmt_num(v_measured['mod_rmse'])}，"
                f"未优于最佳 generated control {fmt_num(v_best_metrics['mod_rmse'])} ({v_best})，"
                "但 measured calibrated 的 stripe leakage 明显低于 generated binary control。"
            ),
            "evidence": ["outputs/forward/t6_hv/model_compare.json"],
            "status": "answered_with_directional_failure",
        },
        {
            "id": 7,
            "question": "加入谐波/网格项后 A_stack 残余网格是否下降？",
            "answer": (
                "没有得到可直接作为形貌结论的稳定下降。固定 forward 的 harmonic/grid/blur rows "
                "对 RMSE 改善很小；aggressive A-only grid penalty 可把 grid energy 从 "
                f"{fmt_num(baseline_row['grid_energy'])} 降到 {fmt_num(aggressive_row['grid_energy'])}，"
                f"但平台 RMS 从 {fmt_num(baseline_row['platform_rms'])} 恶化到 "
                f"{fmt_num(aggressive_row['platform_rms'])}，因此被拒绝。"
            ),
            "evidence": ["outputs/ablation/t6_hv/metrics.json", "outputs/height/platform_compare.json"],
            "status": "answered_negative",
        },
        {
            "id": 8,
            "question": "优化后的 A_stack 是否比 A_cls 更干净？",
            "answer": (
                "取决于判据。A-only 优化在 modulation loss/grid loss 上更低，但过拟合并破坏平台平整度。"
                "最终推荐的 measured-calibrated platform-constrained A_stack 将 modulation loss 从 "
                f"{fmt_num(baseline_row['modulation_loss'])} 降到 {fmt_num(measured_platform_row['modulation_loss'])}，"
                f"同时平台 RMS 保持在 {fmt_num(measured_platform_row['platform_rms'])}，但 grid energy "
                f"{fmt_num(measured_platform_row['grid_energy'])} 未低于 baseline。"
            ),
            "evidence": ["outputs/ablation/t6_hv/metrics.json", "outputs/platform_opt/t6_hv_measured/metrics.json"],
            "status": "answered_with_caveat",
        },
        {
            "id": 9,
            "question": "高度图是否更稳定？",
            "answer": (
                "稳定性门限通过，但不能声明绝对更优。generated platform 早停的高度 std "
                f"{fmt_num(generated_platform_row['height_std'])} 略低于 baseline "
                f"{fmt_num(baseline_row['height_std'])}；measured-calibrated platform 早停 std 为 "
                f"{fmt_num(measured_platform_row['height_std'])}，平台 RMS 为 "
                f"{fmt_num(measured_platform_method['pooled_platform_rms'])}，仍在 1.25x baseline gate 内。"
            ),
            "evidence": ["outputs/height/platform_compare.json", "outputs/ablation/t6_hv/metrics.json"],
            "status": "answered_with_caveat",
        },
        {
            "id": 10,
            "question": "失败点在哪里？",
            "answer": (
                "主要失败点是 V 方向 measured calibrated RMSE 未优于 generated control；"
                "aggressive A-only、theta、affine warp 虽降低重投影损失但平台 RMS 失稳；"
                "`台阶曝光2` 的自动平台 ROI 失败；本地 height-truth 搜索找到 "
                f"{height_truth_candidate_count} 个高度相关文件，但没有独立标定真值；"
                "PyTorch 缺失使 tiny-net/autograd 阶段跳过。"
            ),
            "evidence": [
                "outputs/forward/t6_hv/model_compare.json",
                "outputs/height/platform_compare.json",
                "outputs/complete_samples/t6_hv_measured/metrics.json",
                "outputs/height_truth_assets/assets.json",
                "outputs/audit/coverage.json",
            ],
            "status": "answered",
        },
        {
            "id": 11,
            "question": "是否存在不可辨识或过拟合风险？",
            "answer": (
                "存在。A、M_theta、D0 的尺度和结构残差可互相吸收；A_cls 只能作为弱约束；"
                "低 modulation loss 不等价于可靠高度。当前用 measured DMD/calibration、参数低维化、"
                "平台 RMS gate、complete-sample 对比和审计报告来降低风险；"
                f"{height_truth_pseudo_count} 个本地伪参考资产只能做诊断对比，仍需独立高度真值验证。"
            ),
            "evidence": [
                "outputs/ablation/t6_hv/metrics.json",
                "outputs/height_truth_assets/assets.json",
                "outputs/audit/coverage.json",
            ],
            "status": "answered",
        },
        {
            "id": 12,
            "question": "下一步最值得做什么？",
            "answer": (
                "优先做三件事：第一，针对 V 方向做受限 phase/blur/smooth residual warp 诊断；"
                "第二，先确认本地伪参考资产的来源，或引入独立台阶高度/外部形貌真值来计算绝对误差；"
                "第三，在有 PyTorch 后再做 autograd 版低维 warp/A 优化和 tiny-net，不直接训练大网络。"
            ),
            "evidence": [
                "outputs/final_report.md",
                "outputs/height_truth_assets/height_truth_assets_report.md",
                "outputs/audit/coverage.json",
            ],
            "status": "answered_recommendation",
        },
    ]

    return {
        "artifact_type": "final_required_question_answers",
        "sample": "细台阶",
        "crop": "center_256x256",
        "excluded_samples": excluded,
        "z_unit": z_unit["inferred_z_unit"],
        "z_unit_truth_scope": "local convention, not independent height truth",
        "height_truth_asset_search": {
            "candidate_count": height_truth_candidate_count,
            "pseudo_reference_candidate_count": height_truth_pseudo_count,
            "independent_truth_found": height_truth_found,
        },
        "audit_status": audit["overall"],
        "audit_counts": audit["status_counts"],
        "answers": answers,
    }


def write_markdown(path: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Final Required Research Questions",
        "",
        "## Material Passport",
        "",
        "- Artifact type: final required-question answer appendix",
        f"- Sample: `{data['sample']}`",
        f"- Crop: `{data['crop']}`",
        f"- Excluded samples: `{', '.join(data['excluded_samples'])}`",
        f"- Height unit: `{data['z_unit']}` ({data['z_unit_truth_scope']})",
        (
            "- Height-truth asset search: "
            f"`{data['height_truth_asset_search']['candidate_count']}` candidates, "
            f"`{data['height_truth_asset_search']['pseudo_reference_candidate_count']}` pseudo-references, "
            f"independent truth found `{data['height_truth_asset_search']['independent_truth_found']}`"
        ),
        f"- Audit status: `{data['audit_status']}`; counts `{data['audit_counts']}`",
        "",
        "## Answers",
        "",
    ]
    for item in data["answers"]:
        lines += [
            f"### Q{item['id']}. {item['question']}",
            "",
            f"Status: `{item['status']}`",
            "",
            item["answer"],
            "",
            "Evidence:",
        ]
        lines += [f"- `{path}`" for path in item["evidence"]]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile explicit answers to the final OS-SIM research questions.")
    parser.add_argument("--output", default="outputs/final_questions")
    args = parser.parse_args()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_answers()
    data_io.write_json(out_dir / "answers.json", data)
    data_io.write_json(out_dir / "config.json", vars(args))
    write_markdown(out_dir / "final_question_answers.md", data)
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P13 Final Required Question Answers",
        [
            "- Compiled explicit answers to the 12 required final research questions.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote final question answers to {out_dir}")


if __name__ == "__main__":
    main()
