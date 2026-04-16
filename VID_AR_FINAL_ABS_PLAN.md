# AR Pose Redesign With Multi-Context Motion Prior

**Summary**
- 把当前“利用 context 去 refine target abs pose”的旧逻辑整体替换为新主线：`abs coarse`、`multi-context explicit rel`、`motion trend latent` 三路并行，最终由一个 `final absolute head` 直接重预测 target absolute pose。
- 不再使用硬 `compose` 作为最终输出，也不再让 `rel` 的语义依附于 `abs refine`。`rel` 只表示显式 motion prior，`abs` 只表示单帧 coarse estimate，`final` 是唯一主输出。
- 设计从一开始面向未来多物体，不能针对 `tube` 过拟合；连续对称、离散对称、非对称只在 supervision/canonicalization 层分流，不在主网络结构里特化。
- 允许删除旧接口、旧命名、旧指标和旧 debug 逻辑；新实现优先保持语义清晰和结构简洁，不为兼容旧 refine 逻辑保留多余复杂度。

**Implementation Changes**
- 主模型结构：
  - 保留现有单帧 pose 粗预测路径，明确命名为 `coarse_abs_pose` 或同义新名，不再复用 `abs_rel_refined` 之类旧语义名称。
  - 新建 `multi_context_motion_branch`，输入 3 个 context 的视觉特征、pose code、context 间相对运动，输出两类信息：显式 `relative pose cue` 和 `motion trend latent`。
  - 新建 `final_pose_head`，输入 `target visual token + coarse abs token + explicit rel token + motion latent`，直接输出最终 absolute rotation 和 translation。
  - `final_pose_head` 采用“分路编码 + target-driven fusion + 直接重预测”，不是 concat 后小 MLP，也不是 gated residual update。
- Relative 路径语义：
  - `rel` 定义为“基于多 context 建模得到的 target motion prior”，仍显式输出 relative SE(3)，用于监督、诊断和 fusion。
  - 不再把 `pred_abs(target)` 当成 relative compose anchor；也不再把 `rel` 视为对 `abs` 的 correction。
  - 关闭让 rel branch 直接读取 target abs pose token 的旧耦合逻辑，避免语义退化回 abs-refine。
- Supervision 设计与防冲突：
  - 监督层级固定为：`final absolute supervision` 为主，`coarse abs supervision` 为辅助，`explicit relative supervision` 为辅助，`temporal regularization` 为弱约束。
  - `final absolute` 负责最终精度，不与 `coarse abs` 共享同一语义目标；`coarse abs` 只保证当前帧观测能力，`rel` 只保证 motion prior 可解释。
  - 对 `final` 不施加“必须接近 coarse abs”或“必须等于 rel compose”这类硬一致性约束，只保留弱一致性或 teacher-style regularization，防止不同监督互相拉扯。
  - 对 `rel` 的 rotation/translation loss 与 `final abs` loss 分开记权重，不复用旧 refine loss 名称，避免训练日志误导。
  - 对 symmetry-aware GT 的 canonicalization 只在 `rel supervision` 和 `final/coarse abs supervision` 的 target 构建阶段处理，不在网络前向语义里混入不一致逻辑。
  - 增加 `branch dropout` 或 `path masking`，训练时随机屏蔽 `coarse abs / rel / motion` 某一路，逼 final head 真正使用多路信息，而不是让某一路监督失效。
- 对称物体通用策略：
  - 非对称物体：完整 absolute 和 relative SE(3) 监督。
  - 离散对称物体：基于统一 canonicalization 策略选择监督 GT，再分别监督 coarse/final/rel。
  - 连续对称物体：relative 的不可观测 spin 不作为主监督对象；final 和 coarse 的评估与 PM/ADD-S 仍保持 symmetry-aware。
  - 这些分支全部挂在 `model_info` 驱动的 supervision dispatch 上，不改变主结构接口。
- 旧逻辑清理与重命名：
  - 删除或下线旧的 `abs_rel_refined` 主路径、旧的 context-refine compose 接口、以及依赖旧语义的 loss/config/output 命名。
  - 所有与新语义不符的名字统一改名，例如旧 `refined/abs_rel_refined/context anchor refine` 相关命名全部替换为 `coarse_abs / rel_motion / final_abs / context_memory` 一类新名。
  - 删除仅为旧逻辑服务的中间输出、loss key、config key、fallback 分支和推理选择逻辑；不保留同义双接口。
  - 如果某些旧实现仍需短期保留作 ablation，必须显式归档到 `legacy_*` 命名空间，不与新主逻辑混用。
- 推理逻辑：
  - 第一块 6 帧继续由 coarse/final 单次多帧推理启动，不做块内 rollout。
  - 后续 chunk 使用前一块最终 pose 作为 context pose 输入，但最终 target pose 始终取 `final absolute output`，不是 rel compose 结果。
  - `explicit rel` 只作为 final head 输入和诊断输出，不作为默认最终 pose source。
- 指标与 debug 适配：
  - 删除旧 `abs_rel_refined_compose_gap`、旧 `context refine improvement`、旧 `teacher force rel compose` 等只服务于旧逻辑的指标。
  - 新指标分为四类：`coarse abs quality`、`explicit rel quality`、`final abs quality`、`fusion usage diagnostics`。
  - 新增 fusion 诊断指标：mask 掉某一路后的性能退化、final 相对 coarse 的提升、context 边界 jitter 改善、各路 token/gate 的使用统计。
  - 清理冗长 debug print，只保留能直接判断“哪一路在工作、哪路失效、监督是否冲突”的高价值指标。

**Interfaces / Naming**
- 输出接口统一为：
  - `coarse_abs_rot`, `coarse_abs_trans`
  - `rel_rot`, `rel_trans`
  - `motion_latent` 或同义内部名
  - `final_abs_rot`, `final_abs_trans`
- 配置接口统一为：
  - `COARSE_ABS_*`
  - `REL_MOTION_*`
  - `FINAL_FUSION_*`
  - `LEGACY_*` 仅用于短期 ablation
- 删除或废弃旧接口：
  - `abs_rot_refined`, `abs_trans_refined`
  - `ABS_REL_REFINED_*`
  - 旧 `context-anchor refine` 配置与指标
- 文件落点：
  - [GDRN.py](/mnt/afs/TransparentObjectPose/core/gdrn_modeling/models/GDRN.py) 负责主结构、loss dispatch、命名迁移和旧逻辑下线。
  - [engine.py](/mnt/afs/TransparentObjectPose/core/gdrn_modeling/engine.py) 负责训练调度、日志 key 迁移、旧 debug 指标清理。
  - [inference_vid_ar.py](/mnt/afs/TransparentObjectPose/inference_vid_ar.py) 负责最终输出源切换到 `final_abs`，并清理旧 pose_source 分支。

**Test Plan**
- 先做结构正确性检查：
  - 确认前向输出只保留新语义的 `coarse/rel/final` 三类结果。
  - 确认旧 `abs_rel_refined` 相关接口、配置、日志在主路径中不再被调用。
- 做监督冲突验证：
  - 分别打开和关闭 `coarse abs`、`rel`、`temporal` 辅助监督，观察 `final abs` 是否稳定提升而非互相牵制。
  - 检查不同 loss 的梯度量级和训练曲线，确保 `final abs` 主监督不被辅助项压制。
  - 做 branch dropout ablation，验证 final head 在缺失任一路时仍可工作，同时完整输入时性能最好。
- 做功能性 ablation：
  - `coarse only`
  - `coarse + rel`
  - `coarse + rel + motion latent`
  - `full final fusion`
- 做序列稳定性评估：
  - frame-wise pose error
  - symmetry-aware error
  - translation `xy/z`
  - jitter / acceleration
  - chunk boundary stability
  - long-horizon drift
- 做 fusion 有效性检查：
  - mask 掉 `rel` 后 final 性能应明显下降但不崩溃
  - mask 掉 `coarse abs` 后 final 仍可依赖 target RGB + motion prior 输出合理结果
  - 如果某一路长期被忽略，视为 fusion 设计失败，需要回到结构或 loss 权重调整

**Assumptions**
- 当前主数据仍以 `tube` 为主，但主实现不能把连续对称物体假设写死。
- 新实现优先清晰和简洁，可以主动删除旧 refine 逻辑、旧接口、旧命名和旧指标。
- v1 的唯一最终输出是 `final absolute pose`；`coarse abs` 和 `explicit rel` 都不是默认最终输出。
