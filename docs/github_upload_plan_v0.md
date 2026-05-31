# GitHub 上传计划 v0

更新时间：2026-05-31

当前项目目录还不是 Git 仓库。上传 GitHub 前建议先把 v0 研究状态固化，再初始化仓库。

## 1. 建议上传内容

适合进入 GitHub：

```text
configs/
docs/
scripts/
src/
README.md
requirements.txt
.gitignore
outputs/**/*.md
outputs/**/*.json
outputs/**/*.png
outputs/**/*.html
```

不建议进入 GitHub：

```text
.venv/
原始数据目录
outputs/**/*.npy
outputs/**/*.npz
outputs/**/*.pt
outputs/**/*.pth
outputs/**/*.zip
outputs/**/*.mat
outputs/**/*.bmp
```

原因：

- 原始数据可能体积大，也可能涉及实验记录管理。
- `.npy/.pt` 和模型权重会让 Git 仓库很快膨胀。
- 大文件后续如需管理，建议使用 Git LFS 或单独数据发布平台。

## 2. 初始化仓库命令

在项目根目录执行：

```powershell
git init
git add .gitignore README.md requirements.txt configs docs scripts src
git add outputs/**/*.md outputs/**/*.json outputs/**/*.png outputs/**/*.html
git status
git commit -m "Add v0 OS-SIM neural inverse prototype baseline"
git tag v0-neural-inverse-prototype
```

注意：PowerShell 对 `outputs/**/*.md` 这种 glob 的行为可能和 Git Bash 不同。如果 `git add outputs/**/*.md` 没有按预期工作，可以改用：

```powershell
git add outputs
git status
```

然后确认 `.gitignore` 已经排除了 `.npy/.pt/.zip` 等大文件。

## 3. 连接 GitHub 远程仓库

如果你已经在 GitHub 创建了空仓库，例如：

```text
https://github.com/<your-name>/PhysicsDeepLearning_OSSIM.git
```

执行：

```powershell
git branch -M main
git remote add origin https://github.com/<your-name>/PhysicsDeepLearning_OSSIM.git
git push -u origin main
git push origin v0-neural-inverse-prototype
```

## 4. 如果后续要上传大文件

如果确实需要版本化部分 `.npy`、`.pt` 或数据资产，建议用 Git LFS：

```powershell
git lfs install
git lfs track "*.npy"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Configure Git LFS for experiment artifacts"
```

但 v0 不建议立刻上传全部大文件。

## 5. 推荐 GitHub README 说明

GitHub 首页建议强调：

1. 这是 OS-SIM 神经逆向恢复研究原型。
2. 原始数据不在仓库中，需要本地配置路径。
3. v0 结果不是完整物理标定结论。
4. 当前最重要限制是没有独立高度真值和跨样本验证。
5. 后续路线包括 theta 消融、loss 消融、z-aware 网络和更物理的退化模型。

## 6. 上传前检查清单

- [ ] `git status` 中没有 `.venv/`。
- [ ] 没有原始数据文件。
- [ ] 没有大型 `.npy/.pt/.zip`。
- [ ] `docs/new_conversation_handoff_v0.md` 已存在。
- [ ] `docs/research_v0_baseline.md` 已存在。
- [ ] `outputs/gpt_pro_review_bundle.zip` 不进入 Git 历史。
- [ ] commit 后打 tag：`v0-neural-inverse-prototype`。
