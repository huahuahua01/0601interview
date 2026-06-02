# 深度学习工程师技术面试

**面试时长：60 分钟**

---

## 考察范围

本轮面试主要考察：

- 计算机视觉与深度学习基础能力
- 问题分析与工程化能力
- AI 辅助开发能力
- 项目经验与技术深度
- 代码组织与交付能力

每道题目均包含 **System Design 讨论** 与 **Coding 实操** 环节，请在规定时间内完成所选题目的设计与代码实现。

---

## 面试说明

本次面试包含 **两道 Coding & System Design 题目**，候选人可 **任选其一** 作答。

| 题目 | 文件 | 建议用时 |
|------|------|----------|
| 题目一：新增检测类别（Green Cone） | [task1/color.md](./task1/color.md) | 60 分钟 |
| 题目二：属性识别扩展 | [task2/attribute.md](./task2/attribute.md) | 60 分钟 |

请在开始前告知面试官你的选择，随后围绕所选题目展开讨论。

面试过程中允许使用：

- ChatGPT
- Claude
- Gemini
- Copilot
- 网络搜索
- 官方文档

我们更关注你的：

- 问题拆解能力
- 技术判断能力
- 工程思维
- 学习能力

而不仅仅是最终答案。

---

## 我们重点关注

相比最终答案，我们更关注：

- 是否能够清晰拆解问题
- 是否理解背后的技术原理
- 是否具备工程落地能力
- 是否能够合理使用 AI 工具提高效率
- 是否能够在信息不完整情况下做出合理技术决策
- 是否具备良好的沟通与表达能力

面试过程中请尽量展示你的思考过程。

---

## 已完成的参考实现

本仓库中已补充两类题目的参考实现与说明，便于你对照查看：

### 题目一：新增检测类别（Green Cone）
- `task1/data_color_aug.py`
- `task1/test_data_color_aug.py`

### 题目二：属性识别扩展
- `task2/alert_pipeline.py`
- `task2/test_alert_pipeline.py`
- `task2/attribute_01.md`

### 运行说明

#### 题目一
```bash
python task1/data_color_aug.py
```

#### 题目二测试
```bash
pytest task2/test_alert_pipeline.py -q
```

### 说明
- `task2/alert_pipeline.py` 已包含业务端告警流水线实现
- `task2/test_alert_pipeline.py` 用于演示告警触发、持续报警与 cooldown 行为
- `task2/attribute_01.md` 为题目二的整理版答案

