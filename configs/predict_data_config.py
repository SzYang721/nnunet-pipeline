from pathlib import Path

# 预测相关配置
PREDICT_CONFIGS = {
    "Dataset001_ExampleA":{
        "train_data":{
            "input_path": Path("/path/to/your/input/data"),
            "output_path": Path("/path/to/your/output/data"),
            "model_type": "3d_fullres",  # 或其他有效的 configuration 类型
        },
        "valid_data":{
            "input_path": Path("/path/to/your/valid/data"),
            "output_path": Path("/path/to/your/output/data"),
            "model_type": "3d_fullres",  # 或其他有效的 configuration 类型
        }
    },
    "Dataset002_ExampleB":{
        "train_data":{
            "input_path": Path("/path/to/your/train/data"),
            "output_path": Path("/path/to/your/output/data"),
            "model_type": "3d_fullres",  # 或其他有效的 configuration 类型
        },
        "valid_data":{
            "input_path": Path("/path/to/your/valid/data"),
            "output_path": Path("/path/to/your/output/data"),
            "model_type": "3d_fullres",  # 或其他有效的 configuration 类型
        }
    },
    },
    "Dataset003_ExampleC": {
        "train_data":{
            "input_path": Path("/path/to/your/train/data"),
            "output_path": Path("/path/to/your/output/data"),
            "pred_stage_path": Path("/path/to/your/pred/stage/data"),
            "input_csv_path": "/",
            "model_type": "3d_cascade_fullres",
            "fold": 0,
            "force_all": False,
            "file_type": "nrrd",  # None=自动检测，或指定 "bmp"/"nrrd" 强制使用特定格式
        }
    },
}