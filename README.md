# nnunet-pipeline

基于 **nnU-Net v2** 的 research 训练与推理脚本，支持：

- **训练**：`nnUNetv2_plan_and_preprocess` + `nnUNetv2_train`
- **推理**：自定义推理 / 自动 best configuration（解析 `inference_instructions.txt`，支持 single/ensemble）
- **cascade**：当配置需要时，运行时自动为 `nnUNetv2_predict` 追加 `-prev_stage_predictions`

---

## 快速开始（Fork 后直接运行）

1. **克隆并进入仓库**
   ```bash
   git clone https://github.com/<你的用户名>/nnunet-pipeline.git
   cd nnunet-pipeline
   ```

2. **创建环境并安装依赖**
   ```bash
   conda create -n nnunet_env python=3.11 -y
   conda activate nnunet_env
   pip install -r requirements.txt
   ```
   > PyTorch/CUDA 请按本机版本单独安装（如 `pip install torch` 或 conda）。

3. **安装 nnU-Net v2**（若尚未安装）
   ```bash
   pip install nnunetv2
   ```

4. **配置 nnU-Net 数据路径（必须）**  
   在 `train_with_nnUNet.py` 和 `predict_with_nnUNet.py` 开头修改这三行为你自己的路径：
   ```python
   os.environ['nnUNet_raw'] = "/your/path/to/nnUNet_raw"
   os.environ['nnUNet_preprocessed'] = "/your/path/to/nnUNet_preprocessed"
   os.environ['nnUNet_results'] = "/your/path/to/nnUNet_results"
   ```
   或使用环境变量：运行前 `export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...`。

5. **运行方式**  
   脚本使用相对导入，需以包形式运行。仓库内已包含 `__init__.py`。请将仓库目录命名为合法 Python 包名（如 `nnunet_pipeline`），然后在其**上一级目录**执行：
   ```bash
   # 克隆时可直接指定目录名（避免连字符）
   git clone https://github.com/<你的用户名>/nnunet-pipeline.git nnunet_pipeline
   cd nnunet_pipeline
   # ... 安装依赖、配置路径后 ...
   cd ..
   python -m nnunet_pipeline.train_with_nnUNet -d Dataset038_Spine_Fracture --plan_only
   python -m nnunet_pipeline.predict_with_nnUNet --task Dataset038_Spine_Fracture --input_folder ... --output_folder ... --use_best_config
   ```
   若将本仓库放在 `code/train_code/` 下，则 `cd code` 后执行 `python -m train_code.train_with_nnUNet ...` 即可。

---

## 环境与依赖

### 1) 创建环境

推荐使用你已有的 `nnunet_env`，或新建一个 Python 3.11 环境：

```bash
conda create -n nnunet_env python=3.11 -y
conda activate nnunet_env
```

### 2) 安装依赖

仓库根目录提供 `requirements.txt`（已排除 nvidia-*，便于跨 CUDA 环境安装）。

```bash
pip install -r requirements.txt
```

> PyTorch/CUDA 建议按你机器与 CUDA 版本单独安装（例如使用官方 wheel/conda）。

### 3) 安装/确认 nnU-Net v2

确认已安装 `nnunetv2` 且命令可用：

```bash
nnUNetv2_predict -h
nnUNetv2_train -h
nnUNetv2_find_best_configuration -h
```

---

## nnU-Net 路径（必须）

nnU-Net v2 依赖以下环境变量定位数据与结果目录：

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

**建议**：将上述路径直接写入脚本顶部，便于固定环境、避免每次 export。

- **训练**：在 `train_with_nnUNet.py` 文件开头找到 `os.environ["nnUNet_raw"]` 等三行，改为你自己的路径。
- **推理**：在 `predict_with_nnUNet.py` 文件开头同样修改三处路径。

若希望用环境变量，也可在运行前执行：

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

---

## 配置文件

- `configs/train_config.py`
  - `TRAIN_DATASET_CONFIGS`：每个 dataset 的训练配置（configurations/folds/train_args）
- `configs/predict_data_config.py`
  - `PREDICT_CONFIGS`：每个 dataset 的推理默认路径与参数（input/output/model/fold/file_type 等）

脚本支持：**config 作为默认值，命令行参数覆盖 config**。

---

## 训练（train_with_nnUNet）

入口：`train_with_nnUNet.py`

### 1) 只做 plan & preprocess

```bash
cd /path/to/code   # 或仓库上一级（若以 nnunet_pipeline 包运行）
conda activate nnunet_env

python -m train_code.train_with_nnUNet \
  -d Dataset038_Spine_Fracture \
  --plan_only
```

### 2) 只做训练（跳过 plan）

适用于你已经完成 preprocessing 的场景：

```bash
cd /path/to/code
conda activate nnunet_env

python -m train_code.train_with_nnUNet \
  -d Dataset038_Spine_Fracture \
  --train_only \
  -c 3d_fullres \
  -f 0
```

### 3) 多 fold / 多 trainer / 多 plans

- `-f/--folds` 支持多个 fold
- `-tr/--trainer`、`-p/--plans` 支持多个值，会做 trainer×plans 组合训练

示例：

```bash
python -m train_code.train_with_nnUNet \
  -d Dataset043_Osteosclerosis_small_region \
  --train_only \
  -c 2d \
  -f 0 1 \
  -tr nnUNetTrainer nnUNetTrainerNoMirroring \
  -p nnUNetPlans nnUNetResEncUNetPlans_49G
```

---

## 推理（predict_with_nnUNet）

入口：`predict_with_nnUNet.py`

### 模式 A：best configuration（推荐）

该模式会读取/生成 `nnUNet_results/DatasetXXX_*/inference_instructions.txt`，并按顺序执行：

- one or more `nnUNetv2_predict`
- 如果 best 是 ensemble：额外执行 `nnUNetv2_ensemble`（并自动为 predict 命令补 `--save_probabilities`）
- 最后执行 `nnUNetv2_apply_postprocessing`

```bash
cd /path/to/code
conda activate nnunet_env

python -m train_code.predict_with_nnUNet \
  --task Dataset038_Spine_Fracture \
  --input_folder /path/to/input \
  --output_folder /path/to/output \
  --use_best_config \
  --file_type nrrd \
  --continue_prediction \
  --npp 3 --nps 3
```

### 模式 B：自定义推理（手动指定模型参数）

```bash
python -m train_code.predict_with_nnUNet \
  --task Dataset038_Spine_Fracture \
  --input_folder /path/to/input \
  --output_folder /path/to/output \
  --model 3d_fullres \
  --folds 0 1 2 3 4 \
  --trainer nnUNetTrainer \
  --plans nnUNetPlans \
  --file_type nrrd \
  --npp 3 --nps 3
```

### cascade 推理（需要前一阶段预测）

如果配置为 `3d_cascade_fullres`（或命令需要 prev_stage），请提供 `--prev_stage_predictions`：

```bash
python -m train_code.predict_with_nnUNet \
  --task Dataset038_Spine_Fracture \
  --input_folder /path/to/stage2_input \
  --output_folder /path/to/output \
  --use_best_config \
  --prev_stage_predictions /path/to/stage1_pred \
  --file_type nrrd \
  --npp 3 --nps 3
```

---

## 输入数据格式约定

推理输入目录下需要存在 `*_0000.<ext>` 文件（例如 `case_0000.nrrd`）。

- `--file_type nrrd` 会在输入目录中匹配 `*_0000.nrrd`

---

## 常见问题

### 1) 为什么建议用 `python -m ...`？

本仓库脚本使用了相对导入（例如 `from .utils...`），以包形式运行（`python -m nnunet_pipeline.xxx` 或 `python -m train_code.xxx`）可避免导入失败。

### 2) best configuration 如何决定？

由 `nnUNetv2_find_best_configuration` 基于交叉验证结果（Dice）选择最佳 single/ensemble，并生成 `inference_instructions.txt`。

---

## Citation

若使用本代码或 nnU-Net 进行实验，请引用：

> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature methods*, 18(2), 203-211.

---

## 文件索引

- `train_with_nnUNet.py`：训练入口（plan/preprocess + train）
- `predict_with_nnUNet.py`：推理入口（best config / custom）
- `utils/train_utils.py`：训练命令构建与参数解析
- `utils/predict_utils.py`：instructions 解析、命令构建（ensemble/cascade 支持）
- `utils/run_utils.py`：输出监控进度条
- `configs/train_config.py`：训练配置
- `configs/predict_data_config.py`：推理配置

