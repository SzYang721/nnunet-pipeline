"""
nnUNet 训练配置
train_args: 传递给 nnUNetv2_train 的参数映射，仅当用户只传入 dataset 时使用。
支持的 key: tr, p, pretrained_weights, npz, c, val, val_best, disable_checkpointing, device

tr 和 p 可为 str 或 list。若为 list，将训练 num_trainers * num_plans 种组合。
例如: "tr": ["nnUNetTrainer", "nnUNetTrainerNoMirroring"], "p": ["nnUNetPlans", "nnUNetResEncUNetPlans_49G"]
"""

# 训练配置：dataset 名称 -> 配置字典
TRAIN_DATASET_CONFIGS = {
    "Dataset001_ExampleA": {
        "verify_dataset_integrity": True,
        "configurations": ["3d_fullres"],
        "folds": 0,
        "train_args": {
            "npz": True,
            "device": "cuda",
        },
    },
    "Dataset002_ExampleB": {
        "verify_dataset_integrity": True,
        "configurations": ["3d_fullres"],
        "folds": 0,
        "train_args": {
            "npz": True,
            "device": "cuda",
        },
    },
    "Dataset003_ExampleC": {
        "verify_dataset_integrity": True,
        "configurations": ["2d"],
        "folds": 0,
        "train_args": {
            "npz": True, 
            "device": "cuda"
        },
    },
    "Dataset004_ExampleD": {
        "verify_dataset_integrity": True,
        "configurations": ["3d_fullres","3d_lowres", "3d_cascade_fullres"],
        "folds": [0, 1, 2, 3, 4],
        "train_args": {
            # 探索多组合示例: "tr": ["nnUNetTrainer", "nnUNetTrainerNoMirroring"], "p": ["nnUNetPlans", "nnUNetResEncUNetPlans_49G"]
            "tr": ["nnUNetTrainer", "nnUNetTrainerNoMirroring"],
            "p": ["nnUNetPlans", "nnUNetResEncUNetPlans_49G"],
            "npz": True, 
            "device": "cuda"
        },
    },
}

# 默认要训练的数据集（命令行可覆盖）
DEFAULT_DATASET_NAME = "DatasetXXX_YourDatasetName"

# 5-fold 交叉验证的 fold 列表（命令行可覆盖）
DEFAULT_FOLDS = [0, 1, 2, 3, 4]
