# 题目一：新增检测类别（Green Cone）

---

## 背景

公司目前已有一个基于 YOLO 的安全监控系统。

系统已经能够检测：

- Person
- Helmet
- Vest
- Vehicle
- Cone（Red / Orange，红色 / 橘色锥桶）

并已经完成训练和部署。

---

## 任务

现有模型已能检测常见颜色的锥桶（Cone）：

- Red Cone（红色锥桶）
- Orange Cone（橘色锥桶）

客户提出新需求：

系统需要新增检测以下目标：

- Green Cone（绿色锥桶）

要求：

- 尽可能复用现有系统
- 降低重新训练成本
- 保证上线周期最短

---

### 请回答

#### 1. 数据层面

你会如何收集训练数据？

需要多少数据量？

如何保证数据质量？

如何划分：

- Train
- Validation
- Test

##### Coding 任务：数据增强

在已有红色锥桶数据的基础上，通过颜色变换与几何增强，低成本扩展绿色锥桶训练样本。

**要求：**

1. **数据采集**
  - 使用 data crawler 或手动下载 **10 张** 网络上的锥桶图片，放入 `test_images/` 目录
2. **增强变换**（至少包含以下类型，可扩展）
  - 颜色变换：将红色锥桶转为绿色（复用已有数据集思路）
  - 水平翻转
  - 视角变换（Affine Transform）
  - 尺度缩放
  - 其他你认为有效的变换
3. **实现脚本 `data_color_aug.py`**
  - 输入：`test_images/` 中的测试图片
  - 处理：对每张图片应用颜色变换，并基于增强变换种类 **N** 生成组合
  - 输出：写入 `output_images/`
  - **数量约束**：设增强变换种类数为 **N**，则最终输出图片数量应为 **10 × 2^N**（每张原图对应 2^N 种变换组合）
4. **交付说明**
  - 简要说明颜色变换的实现思路（如 HSV 空间 hue shift、颜色映射等）
  - 说明 2^N 组合的生成逻辑
  - 运行脚本后，`output_images/` 中图片数量符合预期

---

#### 2. 标注层面

如何进行数据标注？

如何制定标注规范？

如何保证多人标注一致性？



---

#### 3. 模型层面

你会选择：

- 从头训练（Train From Scratch）
- Fine-tune
- Continual Learning

中的哪一种方案？

为什么？

---

#### 4. 训练层面

你会调整哪些训练参数？

例如：

- Learning Rate
- Batch Size
- Data Augmentation
- Loss Weight
- Class Weight

为什么？

---

#### 5. 验证层面

如何验证新增类别不会影响已有类别效果？

你会关注哪些指标？

例如：

- Precision
- Recall
- mAP
- F1 Score

---

#### 6. 部署层面

模型训练完成后：

如何部署到生产环境？

如果目标平台是：

- NVIDIA Jetson Orin

你会做哪些优化？

例如：

- TensorRT
- FP16
- INT8
- Batch Optimization

