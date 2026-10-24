# DR-Bridge 模型

本包包含双向交叉注意力、图像监察、临床指标关联监察、自动注意力调整，以及完整串联入口、示例、测试和哈希校验。

## 四个主要代码

| 代码 | 内容 |
| --- | --- |
| `src/drbridge/model.py` | 临床数值、身份及缺失状态编码，FiLM与门控，动态双向交叉注意力，融合与分级 |
| `src/drbridge/image_monitor.py` | 独立ConvNeXt-Tiny单一四分类头，有效证据及三类异常，Top-2连续图像评分 |
| `src/drbridge/indicator_monitor.py` | 头平均注意力、面积覆盖加权、26项完整排名、Top-6参考覆盖核验 |
| `src/drbridge/adjustment.py` | 异常抑制、有效证据增强、8×8到16×16空间回写与重新归一化 |

`pipeline.py`把上述环节连接起来，保留专科复核标记，并通过冻结主模型再次推理输出调整结果。`integrity.py`负责模型状态哈希和检查点哈希。

## 原代码与最终论文的对应

- 主模型的`infer_from_tokens`与`_attention_mask_or_none`沿用作者原源码，AST哈希与原文件哈希记录在`PROVENANCE.json`。
- 公开构造器采用显式配置，保留论文的26项输入和16×16网络空间位置；配置示例中的其他数值用于软件运行，不作为原实验训练参数。
- 图像监察更新为最终的一个四分类头，不沿用旧版二分类加异常类型头。
- Top-2取同一图像最高两项异常概率的均值；仅一项时直接使用该项。
- 指标分支对区域内的26项关联得分排序，Top-6未命中Scr、eGFR、UA、HbA1c及uAlb时形成关联异常标记。
- 待进一步辨别的图像保留初始预测并进入专科复核；指标关联异常保留复核材料，同时允许适用的图像注意力调整。
- 空间抑制与增强可以共同作用，调整前后原图、临床输入和主模型权重不变。

## 去敏范围

包内没有患者影像、标注、个人身份信息、私人目录、训练权重、训练集拟合统计量或未公开的完整训练方案。保密配置通过调用者显式传入，示例配置不冒充真实训练配置。

这是可运行的模型实现，不是完整实验复现包。包内测试与示例验证计算行为，不能替代论文临床结果的独立复现。若需用自有权重运行，应提供匹配的模型配置、临床字段顺序、已拟合预处理与选区结果。

## 运行

安装与GPU环境选择见英文README。仓库根目录下执行：

```bash
python scripts/build_manifest.py --check
python scripts/verify_release.py
python -m unittest discover -s tests -v
python examples/run_demo.py --device cpu
python scripts/smoke_backbones.py --device cpu
```

默认示例使用非患者输入与轻量测试编码器。真实ConvNeXt-Small和ConvNeXt-Tiny的离线前向与完整串联检查由`smoke_backbones.py`执行。具备CUDA版PyTorch时可指定`--device cuda`；本次实际测试设备以`VALIDATION_REPORT.md`为准。

完整调用入口为`drbridge.pipeline.infer_with_joint_monitor`。已经得到四分类概率时可使用`frozen_refinement`。证据框必须是模型输入坐标，临床字段必须与检查点顺序一致；实际检验值应与标准化模型输入分别传入。

## 哈希校验

公开包通过 SHA-256 清单逐文件校验。运行 `python scripts/verify_release.py` 即可确认文件完整未被篡改。哈希仅用于保证代码包完整性，不代表临床有效性。

