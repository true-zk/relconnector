# 本地论文写作环境

本目录以所提供的 SJTUThesis 压缩包为起点。压缩包中的 `sjtuthesis.cls` 为
2024-06-29 的 2.1.3 版，在 TeX Live 2026 下编译封面时报错。因此示例文件已更新为
SJTUThesis `507a8c89ce5dbf9c7784facc3061da6deb6ff054`（2026-03-12），
本地文档类同步为 SJTUTeX 2.3.1。原压缩包仍保存在附件目录。

## VS Code

1. 在 VS Code 中打开 `/home/zk/relconnector/thesis` 文件夹。
2. 安装 LaTeX Workshop 扩展。如果通过 Remote SSH 编辑，应在远程主机上安装该扩展。
3. 打开 `main.tex` 或 `contents` 中的章节文件。章节文件包含指向 `main.tex` 的根文件标记。
4. 保存文件时会按 `.vscode/settings.json` 中的 `latexmk (xelatex)` 配方自动编译；也可以在命令面板执行 `LaTeX Workshop: Build LaTeX project`。
5. 生成的 PDF 为 `main.pdf`，可由 LaTeX Workshop 在 VS Code 标签页中预览。

本机的 TeX Live 2026 安装在 `/home/zk/.local/texlive/2026`。VS Code 配方已指定
工具路径和 PATH；新开的 Bash 终端也会通过 `~/.bashrc` 找到编译工具。

## 命令行

在本目录运行 `make all` 编译，运行 `make clean` 清理中间文件。

当前文档按六章安排：绪论、基础与关键技术、MetaRTL、RelX 设计、RelX 实验、
总结与展望。`contents/intro.tex` 已放入此前写好的绪论；第 2 章中与 MetaRTL
有关的基础和第 3 章 MetaRTL 的方法、实验与分析已写入。RelX 对应章节及全篇
摘要仍保留“待补充”概要。`refs.bib` 已接入绪论与 MetaRTL 章节所引用的文献。

MetaRTL 的原始 LaTeX、结果表与图在 `ref/metartl_source`，该目录作为写作参考
不纳入版本管理；正文所用的原稿图片保存在 `figures/metartl`。原稿没有完整给出
硬件、批大小、随机种子、多次独立重复和离线路径矩阵的时间裁剪细节，正式定稿
前应结合实现与实验记录核对。

正式写作时，在 `setup.tex` 中核对暂定题目并填写真实个人信息，再逐节替换
`contents` 中的占位文字。章节入口和顺序在 `main.tex` 中调整。模板原有的
示例正文、图片、个人经历和成果条目已从本目录移出。
