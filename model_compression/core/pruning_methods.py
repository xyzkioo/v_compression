"""Canonical structured methods shared by the CLI and the workbench."""

STRUCTURED_METHODS = {
    'torch_pruning': (
        '幅值通道剪枝',
        '按通道组的权重范数打分，优先删除分数低的通道。依赖图会同步调整下游层的输入通道和相连分支。\n'
        '打分只需要模型权重，适合做快速试剪。实际删减量受通道对齐和结构依赖限制，可在报告里查看各层删了多少通道。',
    ),
    'taylor': (
        'Taylor 通道剪枝',
        '从带标签的训练集中抽样，反传检测损失。用权重与梯度的一阶近似估计删掉一组通道对损失的影响，优先删分数低的通道。\n'
        'calibration_images 控制抽样数量。样本应覆盖关心的类别和场景，用验证集的 mAP50–95 检查剪后精度。',
    ),
    'lamp': (
        'LAMP 跨层剪枝',
        '先按各层自身的权重幅值分布归一化通道组分数，再让候选层一起分配剪枝预算。容易压缩的层可以删得更多。\n'
        '评分只需要模型权重。channel_sparsity 决定候选层的总预算，lamp_stage_cap 限制每层最多删多少，实际删减量还受通道对齐约束。',
    ),
    'fisher': (
        'Fisher 近似剪枝',
        '逐张标注图片反传检测损失，先对梯度平方，再跨样本汇总并计算通道组分数。正负梯度的影响都会计入敏感度。\n'
        '需要带标签的训练集。逐张反传通常比 Taylor 的批量反传更耗时，calibration_images 可调整评估样本数。',
    ),
    'bn_gamma': (
        '分层 BN-γ（实验）',
        '先用标注训练集对候选层的 BN 缩放系数 γ 做稀疏训练，深层候选的惩罚更强。训练后优先删除 |γ| 小的通道。\n'
        '模型需要有可训练的 BN 层。bn_sparse_epochs 控制剪前稀疏训练轮数，剪后微调由 finetune_epochs 单独控制。',
    ),
    'latency_search': (
        '通道 + 残差块搜索',
        '在候选结构中尝试删通道、跳过残差块或组合两者。逐个评估验证集 mAP50–95，并测量本机 PyTorch 前向延迟。\n'
        '需要带标签的训练集和验证集，耗时随 search_candidates 增加。候选保存后会复测，精度降幅、延迟收益和 P95 都用于验收。',
    ),
    'scale': (
        '模型规格缩放',
        '按 target_scale 从 YOLO 的规格表选择网络并重建，例如从 s 切到 n，再从初始化权重迁移形状匹配的参数。\n'
        '模型 YAML 需要定义 scales。选择更小的规格后，查看参数量和权重匹配比例。规格差异大时，剪后微调有助于恢复精度。',
    ),
}

_ALIASES = dict.fromkeys(('torch-pruning', 'dependency', 'dependency_aware'), 'torch_pruning')
_ALIASES.update(dict.fromkeys(('width_scaling', 'yolo_scale'), 'scale'))


def normalize_structured_method(value: str) -> str:
    name = str(value).strip().lower()
    name = _ALIASES.get(name, name)
    if name not in STRUCTURED_METHODS:
        raise ValueError(f'未知结构化剪枝方法：{value}；可选 {"、".join(STRUCTURED_METHODS)}。')
    return name
