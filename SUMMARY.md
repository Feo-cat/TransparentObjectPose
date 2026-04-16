# Transparent Video Object AR 6D Pose Estimation Summary

## 1. 课题目标与问题定义

这个课题的核心目标，是在视频中对透明物体进行高精度、低抖动、时序连续的 6D pose 估计。目标对象主要包括试管、烧杯、玻璃杯等透明或半透明实验器皿。这类对象有几个典型难点：

- 纹理弱、镜面反射强，单帧 RGB 线索不稳定。
- 轮廓和内部折射会导致 mask、深度、局部特征都容易抖动。
- 很多物体存在旋转对称性，直接监督姿态时容易出现“GT 合法但分支不一致”的学习冲突。
- 单帧绝对位姿容易抖，但纯相对位姿又容易累积漂移。

因此，这个系统并不是把任务看成“逐帧独立 6D pose 回归”，而是把它建模为一个带上下文的 autoregressive 视频位姿估计问题：

- 首个时间窗：一次处理 6 帧，全部作为 target。
- 后续时间窗：使用前一窗最后 3 帧作为 context，再预测新的 3 帧 target。

整个系统的设计理念可以概括为一句话：

> 用强视觉 backbone 提供跨帧语义对应，用 relative pose 学习短时稳定运动，用 absolute pose 维持全局坐标，用 late fusion 把两者结合，再用时序损失和推理端平滑把结果压到“既准又稳”。

## 2. 总体架构

### 2.1 输入组织与时序建模

训练和推理都围绕多视图时间窗展开。每个 batch item 不是单张图，而是一组按时间顺序排列的 views。数据层会把同一实例在多个相邻帧中的裁剪 ROI、相机内参、mask、深度、GT pose 等打包成一个多视图样本，并额外构造：

- `rel_rot_mat[i, j]`：view `j` 相对 view `i` 的相对旋转。
- `rel_trans_mat[i, j]`：view `j` 相对 view `i` 的相对平移。

这样模型在训练时天然就能访问“绝对 pose 监督”和“跨帧相对 pose 监督”两类信号。

### 2.2 视觉编码主干

当前实现虽然配置名里仍保留了 `VGGT_BACKBONE` 这个历史命名，但实际加载的是本地 `dinov3_vits16` 权重。也就是说，系统的主干是一个基于 DINOv3 ViT-S/16 的 patch-level 编码器。

其核心作用是：

- 从透明物体的弱纹理区域中提取更稳定的语义对应关系。
- 为跨帧特征匹配提供比传统 ResNet 更强的局部结构一致性。
- 和后续的 patch/grid、RoPE、Transformer decoder 机制自然配合。

编码后的 token 会进入带 RoPE 的 Transformer blocks，再通过 ROI decoder/head 生成：

- ROI 级别的视觉特征。
- 深度与目标 mask。
- 后续绝对 pose、相对 pose、最终融合 pose 所需的表征。

### 2.3 三条 pose 路径

整个 pose 系统不是单头回归，而是三条路径协同：

#### A. 粗绝对位姿分支

粗绝对分支直接基于 target view 的视觉特征预测：

- 粗旋转 `coarse abs rot`
- 粗平移 `coarse abs trans`

它的优点是：

- 保持全局坐标系一致。
- 不依赖上文预测即可工作。
- 在首个 all-target chunk 中尤为重要。

它的缺点是：

- 容易受透明物体外观扰动影响。
- 单帧回归容易有 jitter。

#### B. 相对位姿分支

相对分支使用三类信息：

- target ROI 的视觉特征
- context ROI 的视觉特征
- context pose token

其中 context pose token 由 `DirectPoseEncoder` 编码。这个模块把 context 的旋转矩阵前两列转成 6D rotation representation，再与 3D translation 拼接成 9D 状态，做高频 Fourier positional encoding 后送入 MLP，得到高维 motion token。

相对分支学习的是 target 相对于 context anchor 的运动，当前主配置下使用：

- `REL_TRANS_MODE = fp_delta`

即：

- 相对旋转：`R_rel = R_target * R_anchor^T`
- 相对平移：`t_rel = t_target - t_anchor`

这使相对平移监督变成相机坐标系下的 additive delta，更接近 tracking 场景中的短时运动建模。

#### C. 最终绝对位姿融合分支

最终位姿不是简单选 coarse 或 relative，而是做 late fusion。融合输入包括：

- coarse absolute pose
- relative pose
- context pose code
- context motion token

最终分支输出：

- `final_abs_rot`
- `final_abs_trans`

这条分支的作用是让网络自己学习：

- 什么时候该更相信 coarse absolute 分支
- 什么时候该更相信 relative motion 分支
- 怎样在“全局准确性”和“局部平滑性”之间做折中

当前实现还支持 fusion path dropout，也就是训练时随机 drop coarse / relative / motion 路径的一部分输入，减少 final head 过度依赖单一信息源。

和旧版本不同的是，当前默认不是“每个 target view 单独采样一次 dropout”，而是：

- 对同一个 chunk 内的所有 target view，共享同一组 coarse / relative / motion keep-mask
- 三条路径彼此仍然独立采样
- 至少保证 coarse / relative / motion 不会三条同时被 drop 掉

这更符合 AR 语义，因为同一 chunk 内所有 target 实际共享同一个 context anchor 和 motion condition。

### 2.4 深度与 mask 辅助头

除了 pose 主线，系统还带有深度头 `DepthHead`，直接对 target view 预测：

- 深度图
- 目标 mask

这部分并不是单纯为了可视化，而是承担两个功能：

- 给透明物体提供更强的几何约束
- 给后续 pose/refiner/可视化提供稳定的观测支持

当前深度监督不仅包括 2D depth regression，还包括：

- mask BCE + Dice
- depth gradient consistency
- 3D 点云误差
- 可选的 dense reprojection pose-flow 约束

### 2.5 推理端 autoregressive 闭环

推理脚本 `inference_vid_ar.py` 已经实现完整的 AR 窗口调度：

- 第一个窗：6 帧全 target
- 后续窗：3 context + 3 target

上一窗已经预测出的最后 3 帧 pose，会被组装成当前窗的 pseudo-GT context anchor 输入模型。这样推理时的分布与训练中的 context 注入机制尽量对齐。

此外，推理端还做了两个重要稳定化步骤：

- 旋转使用 SLERP 做因果平滑
- 平移使用 EMA 做因果平滑

ROI 位置和尺度也会做 `ema` 或 `adaptive_ema` 平滑，减少由于 mask 波动造成的 bbox 漂移。

为了便于排查 final fusion 的训练行为，当前训练可视化视频里还会把每个 target frame 对应的 fusion dropout keep 状态直接标在画面上：

- `c1/r0/m1` 表示 coarse 保留、relative 丢弃、motion 保留
- 这样可以直接检查某段视频里的 final 预测是在依赖哪条路径

## 3. 训练策略

### 3.1 多视图 batch 与 AR 对齐训练

训练时并不是固定只喂一种时序模式，而是在两种模式之间切换：

- `all-target`：整个窗内所有视图都作为 target
- `context-then-target`：前 3 帧是 context，后 3 帧是 target

当前主实验配置中：

- `TRAIN_CONTEXT_THEN_TARGET_PROB = 0.8`

这意味着大部分 batch 会模拟真实 AR 推理场景，但仍保留一部分 all-target 训练，用来增强模型在无上下文时的绝对 pose 能力和首窗表现。

### 3.2 上下文锚点来源

当前系统的一个关键训练点，是 context anchor 不是固定只用 GT，而是支持多种来源：

- `gt_noised`
- `predicted_detach`

当前主配置使用：

- `TRAIN_CONTEXT_ANCHOR_SOURCE = predicted_detach`

也就是说，训练时会把模型自己预测出的 context pose 经过 `detach` 后再注入相对分支/最终融合分支。这很重要，因为推理时真正可用的也是“上一时刻模型预测”，而不是理想 GT。

为了防止 predicted anchor 在训练早期太差、把 relative/fusion 分支带崩，当前配置还引入了混合退火策略：

- `PRED_ANCHOR_GT_NOISE_PROB = 0.4`
- `CONTEXT_ANCHOR_NOISE_PROB = 0.5`
- `CONTEXT_ANCHOR_ROT_DEG_STD = 5.0`
- `CONTEXT_ANCHOR_TRANS_STD = 0.02`

这相当于让网络持续见到：

- 预测锚点
- GT + noise 锚点
- 有误差但可恢复的上下文

从而减少训练/推理分布差异。

### 3.3 对称物体的 canonicalization

透明实验器皿里大量对象都带有离散或连续旋转对称性。如果不处理对称性，训练会出现两个严重问题：

- 同一个物理姿态对应多个合法 GT，损失会互相打架。
- 时序上 GT 可能发生“合法但突然换 spin branch”的跳变，导致模型被错误地惩罚为不平滑。

当前系统已经把对称性处理做到了相对完整的一致化：

#### 绝对 pose 监督

这里现在需要区分两件事：

- `abs head` 自己的 GT 监督，不再强行跟随 context anchor 分支
- `final fusion` 和 `relative` 分支，仍然需要和 context anchor 处在同一个对称分支里

当前 `abs head` 的监督策略是 `closest_pose`：

- 对离散对称对象，从所有合法对称候选里选一个最接近当前预测的 GT clone
- 对连续对称对象，也按“最接近当前预测”的方式确定监督分支

这样做的原因是 `abs head` 本身看不到 context pose。如果强行用 context-anchor-based 分支监督，同一张图会因为不同 iteration 里注入的 context branch 不同而被分配到不同 GT 分支，导致收敛困难。

而在进入 `final fusion` 之前，系统又会把 `coarse_rot_raw` 重新 canonicalize 到最接近 context anchor 的对称分支，再作为 fusion 输入。这个重映射是完全 `detach` 的，所以：

- final fusion 拿到的 coarse cue 与 rel/context cue 分支一致
- 梯度不会通过这条 symmetry remap 路径反向污染 abs head

#### 相对 pose 监督

relative branch 的 GT 不再简单沿用原始 `gt_rel_rot_mat / gt_rel_trans_mat`，而是会基于“实际注入到网络中的 context anchor”重新构造监督目标。这样可以保证：

- relative branch 学的参考系
- final fusion 实际消费的参考系

是同一个。

这是当前系统能真正学稳 relative pose 的关键修复之一。

#### 时序损失监督

当 temporal loss 需要回看 `prev / prev2` pose，而这些 pose 来自完整 GT 序列时，系统也会先按同样的 symmetry anchor 做 canonicalization，再计算速度/加速度项。这样就避免了：

- GT 本身在连续对称轴上任意跳 branch
- temporal loss 错把这种 GT branch 切换当成预测抖动

另外，当前实现还新增了 `ABS_TEMPORAL_CANON=True` 对应的逻辑：

- 在计算 abs-head 的 temporal smooth loss 之前，先把当前预测旋转 canonicalize 到“context-last anchor”所在的对称分支
- 如果 temporal loss 需要使用完整 GT 序列作为 `prev / prev2` fallback，也会用同一个 anchor 做 canonicalization

这样 abs-head 的时序损失里，“预测当前帧”和“GT fallback 上一帧/上两帧”都会落在同一个 spin branch，避免 spin flip 造成假的速度/加速度惩罚。

### 3.4 final head 的训练逻辑

当前实现并不是只监督 coarse abs head，而是显式监督 final fused head。也就是说：

- coarse abs head 有一套自己的 pose loss
- final abs head 会再复制一套同类监督，并加前缀 `loss_final_*`

这样做的目的不是“多算一遍 loss”，而是明确要求 fusion head 自己收敛到独立可用、比 coarse 更平滑的最终 pose。

此外，当 context anchor 来自模型预测时，系统还会额外对 context anchor 本身施加一个绝对 pose 监督：

- `loss_context_anchor_abs`

这有助于防止整个 AR 链条里最前面的 anchor 质量过快退化。

### 3.5 时序平滑约束

训练端当前有两类时序损失实现：

#### A. Temporal Smooth Loss

把 CAD 点在相邻帧 pose 下变换到相机坐标系，然后计算：

- 一阶速度差
- 二阶加速度差

当前主配置里真正启用的是二阶加速度项：

- `TEMPORAL_SMOOTH_A_LW = 5.0`
- `TEMPORAL_SMOOTH_V_LW = 0.0`

这说明当前训练重点更偏向“抑制抖动和突然加速”，而不是单纯压缩位移本身。

#### B. Temporal ADD-S Loss

这是对相邻帧变换后点云做双向最近邻 Chamfer 距离的平滑项，对旋转对称更鲁棒。

当前主配置里：

- `TEMPORAL_ADDS_LW = 0.0`

即代码已实现，但当前实验没有开启。

### 3.6 当前主实验配置快照

以下总结基于 `configs/gdrn/labsim/a6_cPnP_lm13.py` 这一主配置：

| 项目 | 当前设置 |
| --- | --- |
| 数据集 | `labsim_train` / `labsim_test` |
| 对称对象 | `test_tube_rack`, `tube` |
| backbone | DINOv3 ViT-S/16 |
| decoder | 6 层 RoPE + attention decoder |
| rotation repr | `allo_rot6d` |
| 每 GPU batch | 12 |
| 优化器 | Ranger, `lr=1e-4` |
| 训练总 epoch | `800 * 12` |
| AR 训练模式 | `80%` context-then-target, `20%` all-target |
| context / target | `3 / 3` |
| context anchor source | `predicted_detach` |
| anchor source switch iter | `60000` |
| symmetry anchor policy | `SYM_ANCHOR_USE_INJECTED=True` |
| relative translation mode | `fp_delta` |
| final fusion dropout | coarse / rel / motion 各 `0.15`，且默认按 sequence/chunk 共享 keep-mask |
| abs temporal canonicalization | `ABS_TEMPORAL_CANON=True` |
| predicted-anchor fallback | `PRED_ANCHOR_GT_NOISE_PROB=0.4` |
| context anchor noise | `0.5`, rot std `5deg`, trans std `0.02m` |
| MTL uncertainty weighting | 已实现，当前关闭 |
| refiner | 已实现，当前关闭 |
| DZI patch-grid adsorption | 已实现，当前主配置关闭 |
| pose video debug interval | `100` iter |

## 4. 训练损失总览

从代码逻辑上看，训练总目标可以概括为：

```text
L_total
= L_dense_roi
+ L_abs_coarse
+ L_rel
+ L_context_anchor
+ L_final_abs
+ L_depth
```

其中：

- `L_dense_roi`：坐标/区域/mask 等 ROI dense supervision
- `L_abs_coarse`：coarse absolute pose supervision
- `L_rel`：relative pose supervision
- `L_context_anchor`：预测 context anchor 的绝对 pose 监督
- `L_final_abs`：final fused pose 的监督
- `L_depth`：深度与目标 mask 的监督

下面按功能分组说明。

### 4.1 ROI 密集预测损失

| Loss 名称 | 监督对象 | 大致计算 | 当前状态 |
| --- | --- | --- | --- |
| `loss_coor_x` | ROI 内 x 坐标图 | 对可见区域内 `pred_x` 与 `gt_x` 做 L1 或 CE | 启用，`XYZ_LW=1.0` |
| `loss_coor_y` | ROI 内 y 坐标图 | 同上 | 启用，`XYZ_LW=1.0` |
| `loss_coor_z` | ROI 内 z 坐标图 | 同上 | 启用，`XYZ_LW=1.0` |
| `loss_mask` | ROI 目标 mask | `L1` / `BCEWithLogits` / `CE` | 启用，当前沿用 base 配置 `MASK_LW=1.0` |
| `loss_region` | ROI region 分类 | 对 region label 做 CE | 启用，`REGION_LW=0.4` |

说明：

- 当前 base 配置中 `XYZ_LOSS_TYPE = L1`，mask 默认是 `L1`，region 用 `CE`。
- 这些损失主要服务于 ROI 几何表征学习，为后续 pose 回归提供更可分辨的局部结构。

### 4.2 粗绝对位姿损失

| Loss 名称 | 监督对象 | 大致计算 | 当前状态 |
| --- | --- | --- | --- |
| `loss_PM_RT` | 粗绝对 pose 整体 | 将 CAD 点分别用预测 pose、GT pose 变换后，做 point matching | 启用，`PM_LW=1.0` |
| `loss_rot` | 粗绝对旋转 | `angular distance(R_pred, R_gt_canon)` 或旋转矩阵 L2 | 启用，`ROT_LW=1.0` |
| `loss_centroid` | 2D centroid 偏移 | 预测中心与 `gt_trans_ratio[:2]` 的 L1/L2/MSE | 启用，`CENTROID_LW=1.0` |
| `loss_z` | 深度 z | 当 `Z_TYPE=REL` 时监督 `gt_trans_ratio[:,2]`；当 `ABS` 时监督 `gt_trans[:,2]` | 启用，`Z_LW=1.0` |
| `loss_trans_xy` | 平移 xy | `out_trans[:,:2]` 与 `gt_trans[:,:2]` 的 L1/L2/MSE | 启用，`TRANS_LW=1.0` |
| `loss_trans_z` | 平移 z | `out_trans[:,2]` 与 `gt_trans[:,2]` 的 L1/L2/MSE | 启用，`TRANS_LW=1.0` |
| `loss_trans_LPnP` | 直接整体平移 | 若不拆分 xy/z，则对整个位移做回归 | 已实现，当前未走此分支 |
| `loss_bind` | `R^T t` 绑定量 | 对 `R^T t` 做 L1/L2/MSE | 已实现，当前关闭，`BIND_LW=0.0` |

说明：

- 当前配置里 `TRANS_LOSS_DISENTANGLE=True`，所以平移监督会拆成 `loss_trans_xy + loss_trans_z`。
- 对 `abs head` 而言，当前不是使用 context-anchor-based 分支，而是使用 prediction-anchored `closest_pose` 监督，再用于 `loss_rot / loss_trans / PM`。
- 与此同时，喂给 final fusion 的 coarse rotation token 会额外被映射到 context anchor 对齐的分支，但这个映射是 `detach` 的，不会改写 abs head 的训练目标。

### 4.3 Relative pose 损失

| Loss 名称 | 监督对象 | 大致计算 | 当前状态 |
| --- | --- | --- | --- |
| `loss_rot_rel` | 相对旋转 | `angular distance(R_rel_pred, R_rel_gt_canon)` 或 L2 | 强启用，`ROT_REL_LW=40.0` |
| `loss_trans_rel` | 相对平移 | 对 `t_rel_pred` 与 `t_rel_gt` 做 L1/L2/MSE/SmoothL1 | 强启用，`TRANS_REL_LW=10.0` |

当前主配置下 relative translation 的具体形式是：

```text
t_rel = t_target - t_anchor
```

并且先做归一化后再回归：

```text
t_rel_sup = clamp(t_rel / normalizer, -1, 1)
pred = tanh(head_output)
loss_trans_rel = SmoothL1(pred, t_rel_sup)
```

其中当前 normalizer 为：

```text
[0.12, 0.07, 0.15]
```

这部分设计的意图是让 relative translation 大多落在 tanh 的近线性区，提高短时位移学习稳定性。

补充说明：

- 连续对称物体的 relative GT 也会做 anchor-aware canonicalization。
- 代码里还有 `loss_rot_rel_axis`、`loss_rot_rel_spin` 等 debug 指标，但它们当前主要用于观测，不直接计入总损失。

### 4.4 Context anchor 与 final fused pose 损失

| Loss 名称 | 监督对象 | 大致计算 | 当前状态 |
| --- | --- | --- | --- |
| `loss_context_anchor_abs` | 预测 context anchor 本身 | 对 predicted context pose 复用一套绝对 pose 损失后求和 | 启用，`CONTEXT_ANCHOR_ABS_LW=1.0` |
| `loss_final_rot` | final 旋转 | final fused pose 上的 `loss_rot` 镜像版本 | 启用，`FINAL_ABS_SUP_LW=1.0` |
| `loss_final_trans_xy` | final 平移 xy | final fused pose 上的平移镜像监督 | 启用 |
| `loss_final_trans_z` | final 平移 z | final fused pose 上的平移镜像监督 | 启用 |
| `loss_final_PM_RT` | final pose point matching | final fused pose 上的 PM 监督 | 启用 |
| `loss_final_temporal_v` | final 一阶时序平滑 | final pose 上的一阶 temporal loss 镜像版本 | 已实现，当前权重为 0 |
| `loss_final_temporal_a` | final 二阶时序平滑 | final pose 上的二阶 temporal loss 镜像版本 | 启用，等价受 `FINAL_ABS_SUP_LW * TEMPORAL_SMOOTH_A_LW` 影响 |
| `loss_final_temporal_adds` | final ADD-S 平滑 | final pose 上的 temporal ADD-S 镜像版本 | 已实现，当前关闭 |

说明：

- `loss_final_*` 本质上不是全新定义，而是把 coarse abs pose 的那套监督再复制到 `final_abs_pose` 上。
- 这能强制 final head 真正承担“最终输出位姿”的职责，而不是只做一个弱辅助分支。
- final branch 实际消费的 coarse rotation 输入，已经先被 canonicalize 到与 context anchor 一致的对称分支，因此 fusion 输入里的 coarse / rel / context 三路语义是对齐的。
- 如果开启了 final fusion path dropout，当前默认也是按 sequence/chunk 共享 keep-mask，而不是对每个 target view 单独随机采样。

### 4.5 时序损失

| Loss 名称 | 监督对象 | 大致计算 | 当前状态 |
| --- | --- | --- | --- |
| `loss_temporal_v` | 一阶速度平滑 | 将 CAD 点变换到相机系后，比较 `pts_t` 与 `pts_prev` | 已实现，当前 `V_LW=0.0` |
| `loss_temporal_a` | 二阶加速度平滑 | 比较 `(pts_t - pts_prev)` 与 `(pts_prev - pts_prev2)` | 启用，`A_LW=5.0` |
| `loss_temporal_adds` | 对称鲁棒时序平滑 | 对相邻帧变换后的点云做双向最近邻 Chamfer | 已实现，当前 `ADDS_LW=0.0` |

说明：

- 当前最核心的时序监督是二阶加速度项，因为它直接惩罚“抖一下、又抖回来”的高频不稳定。
- coarse 分支和 final 分支都可以承受时序约束。
- 在当前主配置里，`ABS_TEMPORAL_CANON=True`，所以 abs-head 的 temporal loss 前会先把预测旋转对齐到 context-last 的对称分支；GT fallback 的 `prev / prev2` 也会用同一 anchor 做 canonicalization。

### 4.6 深度与 mask 损失

| Loss 名称 | 监督对象 | 大致计算 | 当前状态 |
| --- | --- | --- | --- |
| `loss_obj_mask` | target view 目标 mask | `lambda_bce * BCE + lambda_dice * Dice` | 启用，`1.0 + 1.0` |
| `loss_dp_reg` | 深度值本身 | 仅在有效 mask 内做 SmoothL1 depth regression | 启用，`LAMBDA_DP_REG=1.0` |
| `loss_dp_gd` | 深度梯度 | 对 x/y 方向有限差分梯度做 L1 一致性约束 | 强启用，`LAMBDA_DP_GD=50.0` |
| `loss_dp_3d` | 3D 点云几何 | 用相机内参把深度图反投影成 3D 点，比较点坐标差 | 启用，`LAMBDA_DP_3D=5.0` |
| `loss_pose_flow` | 稠密重投影位姿误差 | 用预测 pose 与 GT pose 诱导像素重投影位移，再做 SmoothL1 | 已实现，当前关闭，`LAMBDA_REPROJ=0.0` |

说明：

- `loss_dp_reg` 和 `loss_dp_gd` 更偏向“深度图像质量”。
- `loss_dp_3d` 更偏向“物理尺度下的几何一致性”。
- `loss_pose_flow` 的梯度被刻意设计为不回流到深度预测主干的关键路径上，目的是让深度和姿态各自承担更明确的学习职责。

### 4.7 多任务不确定性加权

代码里已经实现了 uncertainty-based multi-task weighting：

```text
L'_i = L_i * exp(-s_i) + log(1 + exp(s_i))
```

其中 `s_i` 是可学习的 `log_var_i`。这一机制已经覆盖了：

- 绝对 pose loss
- relative pose loss
- depth loss
- temporal loss

但当前 base 配置：

- `USE_MTL = False`

因此当前主实验仍然使用手工设定的 loss 权重。

## 5. 当前系统现状判断

### 5.1 已经比较扎实的部分

从现有实现看，这个项目已经不再是“单帧 pose baseline 加一点时序 trick”，而是形成了比较完整的视频透明物体 AR pose 框架。当前已经比较成熟的能力包括：

- 多视图时间窗数据组织已经完整打通。
- 训练和推理的 AR 窗口机制已经基本对齐。
- 相对位姿分支、context pose encoder、final fusion head 已经实装并接入训练。
- 对称性 canonicalization 已经扩展到绝对 pose、relative pose、final fusion 输入对齐、时间平滑监督四条主线。
- 推理端已经具备 causal pose smoothing 和 ROI smoothing。
- 深度/mask 头已经不仅是辅助可视化，而是进入了训练闭环。
- 训练调试视频已经能直接显示 final fusion 每帧的 coarse / rel / motion keep 状态，便于定位“当前预测主要依赖哪条路径”。

### 5.2 这个系统当前最重要的优势

和很多只靠单帧回归的 6D pose 方法相比，这个系统最核心的优势有三点：

#### A. relative pose 与 absolute pose 的职责分工清楚

- absolute branch 负责全局对齐
- relative branch 负责短时稳定运动
- final fusion 负责两者整合

这比“直接在单帧头上强行加 temporal loss”要更合理。

#### B. 对称性问题处理得比一般实现更彻底

很多系统只在 eval 时考虑对称性，训练时仍然拿原始 GT 硬回归。当前实现已经把 symmetry canonicalization 放进了：

- coarse abs supervision（以 `closest_pose` 方式稳定 abs head）
- relative supervision
- coarse-to-final fusion 输入对齐
- temporal supervision

这对透明器皿这类连续旋转对称对象尤其关键。

#### C. 训练分布和推理分布更接近

当前训练不是只喂 GT context，而是主动让模型适应 predicted/noised context anchor。这一点直接关系到 AR 推理时是否会“第一窗还行，越往后越漂”。

#### D. 最终融合正则化更贴近 AR 语义

当前 final fusion dropout 已经改成“同一 sequence/chunk 内共享 keep-mask”，而不是逐 target 随机。这样虽然少了一点泛化型噪声，但更符合真实 AR 条件，因为同一 chunk 的所有 target 确实共享同一个注入 context。

### 5.3 当前仍然值得关注的风险点

虽然整体框架已经成型，但从实现和配置上看，现阶段仍有几个值得持续关注的方向：

#### A. 透明物体的瓶颈仍然很可能在观测质量

无论 relative/fusion 设计多强，如果：

- mask 边界不稳
- 深度质量不稳
- ROI 漂移较大

最终 pose 仍会受限。因此推理端 external mask、mask clean、ROI smoothing 依然很关键。

#### B. 当前时序监督更偏“抑制加速度”，还不是全套动态建模

目前真正开启的是 `loss_temporal_a`，这对抑制 jitter 很有效，但它本质上还是一个启发式平滑项，不等价于显式的动力学模型或 uncertainty-aware temporal filtering。

#### C. 深度头和 pose 主干之间的协同还有继续优化空间

现在深度头已经很强，但 `loss_pose_flow` 默认关闭，refiner 也默认关闭。这说明系统当前主线仍以“主网络直接出稳定 pose”为主，后续若要进一步抬高上限，可以继续研究：

- 深度对 pose 的更强耦合方式
- render-and-compare refiner 的启用时机
- 推理端 mask/depth/pose 的闭环优化

#### D. 现有 canonicalization 逻辑对表示和对象对称形式仍有前提

当前 coarse-token 到 final-fusion 的 branch remap，本质上是把旋转矩阵重新编码回 `rot6d raw`，所以它目前只对 `ego_rot6d` / `allo_rot6d` 完整成立。主配置现在正好使用 `allo_rot6d`，因此没有问题；但如果后续切到 `quat` 或 `lie_vec`，必须同步扩展这段重编码逻辑。

另外，当前只 remap 了 `coarse_rot_raw`，没有同步 remap `coarse_t_raw`。这在现有 labsim 对象上是安全的，因为当前对称变换不携带平移项；如果未来引入带非零 `sym_trans` 或连续对称 `offset != 0` 的物体，就必须把 coarse translation 一并 canonicalize，否则 final fusion 会收到“rot 已换分支、trans 仍在原分支”的不一致输入。

## 6. 一句话总结当前课题状态

当前这个课题已经从“透明物体单帧 6D pose”推进到了“带相对运动建模、对称性一致化、final late fusion、深度辅助监督、AR 推理平滑”的系统化阶段。就代码状态而言，最关键的问题已经不再是“有没有时序能力”，而是“如何继续把观测质量、relative/absolute 融合质量、以及长序列稳定性一起推到更高水平”。
