CONFIG_ORDER = ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]


def sort_configurations(configs):
    """按 cascade 依赖排序：3d_lowres 在 3d_cascade_fullres 之前。"""
    order_map = {c: i for i, c in enumerate(CONFIG_ORDER)}
    return sorted(configs, key=lambda c: (order_map.get(c, 999), c))


def normalize_to_list(x):
    """将 str 转为 [str]，list 原样返回。"""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [str(x)]


def get_trainer_plan_combinations(trainers, plans):
    """返回 (trainer, plan) 组合列表，num_trainers * num_plans。"""
    tr_list = normalize_to_list(trainers) or ["nnUNetTrainer"]
    p_list = normalize_to_list(plans) or ["nnUNetPlans"]
    return [(tr, p) for tr in tr_list for p in p_list]


def build_train_cmd(dataset_id, configuration, fold, train_args, trainer=None, plan=None):
    """根据 train_args 构建 nnUNetv2_train 命令列表。只传 dataset id。"""
    cmd = ["nnUNetv2_train", str(dataset_id), configuration, str(fold)]
    args = train_args or {}
    tr = trainer if trainer is not None else args.get("tr")
    p = plan if plan is not None else args.get("p")
    if tr:
        cmd.extend(["-tr", str(tr)])
    if p:
        cmd.extend(["-p", str(p)])
    if args.get("pretrained_weights"):
        cmd.extend(["-pretrained_weights", str(args["pretrained_weights"])])
    if args.get("npz"):
        cmd.append("--npz")
    if args.get("c"):
        cmd.append("--c")
    if args.get("val"):
        cmd.append("--val")
    if args.get("val_best"):
        cmd.append("--val_best")
    if args.get("disable_checkpointing"):
        cmd.append("--disable_checkpointing")
    device = args.get("device", "cuda")
    cmd.extend(["-device", str(device)])
    return cmd

def parse_train_args_from_cli(args):
    """从 argparse 结果构建 train_args 映射（与 nnUNetv2_train 参数对应）。tr/p 可为 list。"""
    train_args = {}
    if args.trainer is not None:
        train_args["tr"] = args.trainer  # nargs="+" 时为 list
    if args.plans is not None:
        train_args["p"] = args.plans  # nargs="+" 时为 list
    if args.pretrained_weights is not None:
        train_args["pretrained_weights"] = args.pretrained_weights
    if args.continue_train:
        train_args["c"] = True
    if args.val_only:
        train_args["val"] = True
    if args.val_best:
        train_args["val_best"] = True
    if args.disable_checkpointing:
        train_args["disable_checkpointing"] = True
    if args.device is not None:
        train_args["device"] = args.device
    return train_args