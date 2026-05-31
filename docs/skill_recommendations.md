# OS-SIM 项目可用 Skill 建议

## 结论

当前没有发现已经安装或 OpenAI curated 列表中现成适配“反射式 OS-SIM 三维形貌恢复”的专用 skill。这个项目强依赖你的采集帧序号、点阵扫描语义、OS-SIM 相移规则和后续标签生成方式，因此更适合后续创建一个项目专用 skill。

## 当前可直接使用的 Skill

| Skill | 在本项目中的用途 |
|---|---|
| `academic-research-suite` | 梳理研究问题、实验方案、文献综述、论文结构和审稿式检查 |
| `skill-creator` | 后续创建项目专用 `ossim-3d-reconstruction` skill |
| `skill-installer` | 安装通用辅助 skill，例如 notebook 或 PDF 处理 |
| `latex:latex-compile` | 后续编译论文 LaTeX 稿件 |
| `zotero:Zotero` | 管理文献、导出 BibTeX、维护引用 |
| `hugging-face:huggingface-jobs` | 本地算力不足时，用 Hugging Face Jobs 跑 GPU 训练任务 |
| `hugging-face:huggingface-trackio` | 记录训练指标、实验曲线和模型对比结果 |

## 可考虑安装的通用 Skill

我检查了 OpenAI curated skills 列表，比较相关的候选只有以下两个：

| 可安装 skill | 作用 | 优先级 |
|---|---|---|
| `jupyter-notebook` | 交互式检查数据、可视化 z-stack、快速验证 baseline | 高 |
| `pdf` | 阅读和抽取显微成像、深度学习重建相关论文内容 | 中 |

curated 列表中没有发现 microscopy、SIM、OS-SIM、image reconstruction 或 scientific 3D reconstruction 的专用 skill。实验性 skill 路径当前不可用。

## 建议创建的项目专用 Skill

建议在完成第一版数据读取器和 baseline 后，创建一个名为 `ossim-3d-reconstruction` 的专用 skill。

建议内容：

- `SKILL.md`：记录 OS-SIM 数据读取、点阵标签生成、baseline、训练和评价的固定流程。
- `references/data_schema.md`：记录样本目录结构、z 轴解析、帧序号含义和缺帧处理规则。
- `references/reconstruction_protocol.md`：记录点阵扫描生成高度标签的方法、OS-SIM 输入组合、训练/验证划分和评价指标。
- `scripts/inspect_dataset.py`：自动检查数据完整性并生成 manifest。
- `scripts/make_preview.py`：快速生成指定样本和 z 层的预览图。

这个项目专用 skill 会比通用视觉 skill 更有价值，因为这里最关键的知识不是普通图像模型，而是采集流程和帧语义。
