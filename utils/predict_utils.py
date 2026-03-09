import os
import re
import logging
import shlex
from pathlib import Path
from typing import List, Optional, Tuple
import shutil
import tempfile

# 有效的 nnUNet configuration 类型（用于过滤）
VALID_CONFIGURATIONS = ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]

# =============================================================================
# 解析 dataset 名称
# =============================================================================

def get_dataset_name(task_name: str) -> str:
    """
    将 task_name 转为 nnUNet 规范 dataset 名称（DatasetXXX_Name 格式）。
    若为纯数字（如 38）或 Dataset038 等不完整格式，则从 nnUNet_results 中查找 Dataset038_ 开头的文件夹获取完整名称。
    """
    task_name = str(task_name).strip()
    if task_name.startswith("Dataset") and "_" in task_name and re.match(r"^Dataset\d{3}_.+", task_name, re.I):
        return task_name
    dataset_id = None
    if task_name.isdigit():
        dataset_id = int(task_name)
    else:
        m = re.match(r"^Dataset(\d+)$", task_name, re.IGNORECASE)
        if m:
            dataset_id = int(m.group(1))
    if dataset_id is not None:
        prefix = f"Dataset{dataset_id:03d}_"
        nnunet_results = os.environ.get("nnUNet_results")
        if nnunet_results:
            results_dir = Path(nnunet_results)
            if results_dir.exists():
                candidates = [d.name for d in results_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
                if len(candidates) == 1:
                    return candidates[0]
                if len(candidates) > 1:
                    logging.warning("多个 dataset 匹配 id %s，使用第一个: %s", task_name, candidates[0])
                    return candidates[0]
        return f"Dataset{dataset_id:03d}"
    return task_name


# =============================================================================
# 解析已训练的模型文件夹
# =============================================================================


# nnUNet find_best_configuration / ensemble 要求：5 折齐全且每折 validation 下有 .npz（训练时需 --npz）
REQUIRED_FOLDS = [0, 1, 2, 3, 4]


def discover_trained_configurations(dataset_name_or_id: str) -> List[str]:
    """
    从 nnUNet_results/DatasetXXX/ 扫描已训练的模型文件夹（Trainer__Plans__Configuration 格式），
    提取 configuration 部分，仅保留 2d、3d_fullres、3d_lowres、3d_cascade_fullres。

    与 nnUNet 官方要求一致：仅当该 configuration 对应的训练目录满足以下条件时才计入：
    - 存在完整 5 折结构：fold_0, fold_1, fold_2, fold_3, fold_4；
    - 每个 fold_X/validation 下至少有一个 .npz 文件（训练时需使用 --npz）。
    跳过 ensemble___ 前缀的文件夹。
    """
    nnunet_results = os.environ.get("nnUNet_results")
    if not nnunet_results:
        return []

    dataset_name = get_dataset_name(dataset_name_or_id)
    dataset_dir = Path(nnunet_results) / dataset_name
    if not dataset_dir.exists():
        return []

    configs = set()
    for item in dataset_dir.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if name.startswith("ensemble___"):
            continue
        # 解析出该文件夹对应的 configuration（取名称中匹配的第一个有效配置）
        matched_cfg = None
        for valid_cfg in VALID_CONFIGURATIONS:
            if valid_cfg in name:
                matched_cfg = valid_cfg
                break
        if matched_cfg is None:
            continue
        # 要求 5 折齐全，且每折 validation 下均有 .npz
        try:
            for f in REQUIRED_FOLDS:
                val_dir = item / f"fold_{f}" / "validation"
                if not val_dir.is_dir():
                    break
                npz_files = list(val_dir.glob("*.npz"))
                if not npz_files:
                    break
            else:
                # 所有 5 折均满足条件
                configs.add(matched_cfg)
        except OSError:
            continue

    return sorted(configs, key=lambda c: VALID_CONFIGURATIONS.index(c) if c in VALID_CONFIGURATIONS else 999)


# =============================================================================
# 解析 inference_instructions.txt（Step 1 的输出，供 Step 2/3 使用）
# =============================================================================


def parse_inference_instructions(
    task_name: str,
) -> Tuple[Optional[List[List[str]]], Optional[List[str]], Optional[List[str]]]:
    """
    从 inference_instructions.txt 解析推理和后处理命令模板。
    文件路径: nnUNet_results/DatasetXXX_Name/inference_instructions.txt
    也尝试 inference_instruction.txt（无 s）以兼容。

    返回:
        (inference_cmd_parts_list, ensemble_cmd_parts, postprocessing_cmd_parts)
        - inference_cmd_parts_list: 一个或多个 nnUNetv2_predict 命令的 shlex 解析结果（占位符未替换）
        - ensemble_cmd_parts: nnUNetv2_ensemble 命令（若存在）
        - postprocessing_cmd_parts: nnUNetv2_apply_postprocessing 命令
        若解析失败或文件不存在，返回 (None, None, None)。
    """
    nnunet_results = os.environ.get("nnUNet_results")
    if not nnunet_results:
        logging.warning("nnUNet_results 未设置，无法解析 inference_instructions")
        return None, None, None

    dataset_name = get_dataset_name(task_name)
    candidates = [
        Path(nnunet_results) / dataset_name / "inference_instructions.txt",
        Path(nnunet_results) / dataset_name / "inference_instruction.txt",
    ]
    instructions_path = None
    for p in candidates:
        if p.exists():
            instructions_path = p
            break

    if not instructions_path or not instructions_path.exists():
        logging.warning(f"未找到 inference_instructions.txt: {candidates}")
        return None, None, None

    content = instructions_path.read_text(encoding="utf-8", errors="replace")
    inference_cmds = []
    ensemble_cmd = None
    postprocessing_cmd = None

    # 收集所有 nnUNetv2_predict 行（单模型/ensemble 都可支持）
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("nnUNetv2_predict "):
            inference_cmds.append(line)

    # 查找 nnUNetv2_ensemble 行（可能不存在）
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("nnUNetv2_ensemble "):
            ensemble_cmd = line
            break

    # 查找 nnUNetv2_apply_postprocessing 行
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("nnUNetv2_apply_postprocessing "):
            postprocessing_cmd = line
            break

    if not inference_cmds:
        logging.warning("inference_instructions.txt 中未找到 nnUNetv2_predict 命令")
        return None, None, None

    try:
        inference_parts_list = [shlex.split(cmd) for cmd in inference_cmds]
    except Exception as e:
        logging.warning(f"解析推理命令失败: {e}")
        return None, None, None

    ensemble_parts = None
    if ensemble_cmd:
        try:
            ensemble_parts = shlex.split(ensemble_cmd)
        except Exception as e:
            logging.warning(f"解析集成命令失败: {e}")
            return None, None, None

    pp_parts = None
    if postprocessing_cmd:
        try:
            pp_parts = shlex.split(postprocessing_cmd)
        except Exception as e:
            logging.warning(f"解析后处理命令失败: {e}")
            # pp_parts 保持 None，主流程将不执行后处理

    return inference_parts_list, ensemble_parts, pp_parts


def replace_instruction_placeholders(parts: List[str], replacement_map: dict) -> List[str]:
    """
    将命令中的占位符替换为实际路径。
    例如 INPUT_FOLDER、OUTPUT_FOLDER、OUTPUT_FOLDER_MODEL_1 等。
    """
    result = []
    for p in parts:
        if p in replacement_map:
            result.append(str(replacement_map[p]))
        else:
            result.append(p)
    return result


def build_postprocessing_cmd_from_instructions(
    pp_parts: List[str],
    output_folder: str,
    output_folder_pp: str,
) -> List[str]:
    """
    将后处理命令中的 OUTPUT_FOLDER、OUTPUT_FOLDER_PP 替换为实际路径。
    """
    return replace_instruction_placeholders(
        pp_parts,
        {
            "OUTPUT_FOLDER": str(output_folder),
            "OUTPUT_FOLDER_PP": str(output_folder_pp),
        },
    )


def get_cmd_option_value(cmd_parts: List[str], option: str) -> Optional[str]:
    """获取命令中某个 option 的值，不存在则返回 None。"""
    if option not in cmd_parts:
        return None
    idx = cmd_parts.index(option)
    if idx + 1 >= len(cmd_parts):
        return None
    return cmd_parts[idx + 1]


def set_or_append_cmd_option(cmd_parts: List[str], option: str, value: str) -> List[str]:
    """设置命令中的 option；若不存在则追加。"""
    result = list(cmd_parts)
    if option in result:
        idx = result.index(option)
        if idx + 1 < len(result):
            result[idx + 1] = str(value)
        else:
            result.append(str(value))
    else:
        result.extend([option, str(value)])
    return result


def append_flag_if_missing(cmd_parts: List[str], flag: str) -> List[str]:
    """若 flag 不存在则追加。"""
    result = list(cmd_parts)
    if flag not in result:
        result.append(flag)
    return result


def is_predict_command(cmd_parts: List[str]) -> bool:
    return len(cmd_parts) > 0 and cmd_parts[0] == "nnUNetv2_predict"


def predict_cmd_requires_prev_stage(cmd_parts: List[str]) -> bool:
    """判断该 predict 命令是否需要前一阶段预测。"""
    config = get_cmd_option_value(cmd_parts, "-c")
    return (
        config == "3d_cascade_fullres"
        or "-prev_stage_predictions" in cmd_parts
        or "OUTPUT_FOLDER_PREV_STAGE" in cmd_parts
    )


def build_best_config_base_commands(
    inference_parts_list: List[List[str]],
    ensemble_parts: Optional[List[str]],
    input_folder: Path,
    output_folder: Path,
    prev_stage_runtime_path: Optional[str],
) -> Tuple[List[List[str]], List[Path]]:
    """
    基于 inference_instructions.txt 构建基础命令。
    这里只做占位符替换，不追加运行期参数。
    OUTPUT_FOLDER_MODEL_1 / OUTPUT_FOLDER_MODEL_2 等映射到 output_folder 下固定子目录
    (_model_1, _model_2)，以便跨次运行时 --continue_prediction 对中间结果也生效；
    其他未知 OUTPUT_FOLDER* 仍用临时目录并在调用方清理。
    """
    replacement_map = {
        "INPUT_FOLDER": str(input_folder),
        "OUTPUT_FOLDER": str(output_folder),
    }
    if prev_stage_runtime_path:
        replacement_map["OUTPUT_FOLDER_PREV_STAGE"] = str(prev_stage_runtime_path)

    extra_tmp_dirs = []
    # nnUNet find_best_configuration 生成的 ensemble 占位符：OUTPUT_FOLDER_MODEL_1, OUTPUT_FOLDER_MODEL_2, ...
    model_placeholder_re = re.compile(r"^OUTPUT_FOLDER_MODEL_(\d+)$")

    def ensure_output_placeholder(placeholder: str):
        if placeholder in replacement_map:
            return
        m = model_placeholder_re.match(placeholder)
        if m:
            replacement_map[placeholder] = str(output_folder / f"_model_{m.group(1)}")
            return
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"nnunet_{placeholder.lower()}_"))
        replacement_map[placeholder] = str(tmp_dir)
        extra_tmp_dirs.append(tmp_dir)

    for cmd_parts in inference_parts_list:
        for token in cmd_parts:
            if token.startswith("OUTPUT_FOLDER") and token not in ("OUTPUT_FOLDER_PP", "OUTPUT_FOLDER_PREV_STAGE"):
                ensure_output_placeholder(token)
    if ensemble_parts is not None:
        for token in ensemble_parts:
            if token.startswith("OUTPUT_FOLDER") and token not in ("OUTPUT_FOLDER_PP", "OUTPUT_FOLDER_PREV_STAGE"):
                ensure_output_placeholder(token)

    base_cmds = [replace_instruction_placeholders(cmd_parts, replacement_map) for cmd_parts in inference_parts_list]
    if ensemble_parts is not None:
        base_cmds.append(replace_instruction_placeholders(ensemble_parts, replacement_map))
    return base_cmds, extra_tmp_dirs


def build_custom_base_command(
    task_name: str,
    model_type: str,
    trainer: str,
    plans: str,
    fold: int,
    folds: Optional[List[int]],
    input_folder: Path,
    output_folder: Path,
) -> List[str]:
    """构建自定义模式下的基础 predict 命令。"""
    cmd = [
        "nnUNetv2_predict",
        "-d", str(task_name),
        "-i", str(input_folder),
        "-o", str(output_folder),
        "-c", str(model_type),
        "-tr", trainer,
        "-p", plans,
    ]
    if folds is not None and isinstance(folds, (list, tuple)):
        cmd.extend(["-f"] + [str(f) for f in folds])
    else:
        cmd.extend(["-f", str(fold)])
    return cmd


def append_runtime_predict_options(
    cmd_parts: List[str],
    prev_stage_runtime_path: Optional[str],
    continue_prediction: bool,
    num_processes_preprocessing: int,
    num_processes_saving: int,
    require_probabilities: bool,
) -> List[str]:
    """
    给 predict 命令统一追加运行期参数。
    instructions 只提供基础模板，运行期参数在这里补齐。
    """
    if not is_predict_command(cmd_parts):
        return list(cmd_parts)

    result = list(cmd_parts)
    result = set_or_append_cmd_option(result, "-device", "cuda")
    if require_probabilities:
        result = append_flag_if_missing(result, "--save_probabilities")
    if continue_prediction:
        result = append_flag_if_missing(result, "--continue_prediction")
    if num_processes_preprocessing > 0:
        result = set_or_append_cmd_option(result, "-npp", str(num_processes_preprocessing))
    if num_processes_saving > 0:
        result = set_or_append_cmd_option(result, "-nps", str(num_processes_saving))

    if predict_cmd_requires_prev_stage(result):
        if prev_stage_runtime_path:
            result = set_or_append_cmd_option(result, "-prev_stage_predictions", str(prev_stage_runtime_path))
        else:
            logging.warning("检测到 cascade 推理命令，但当前病例未提供可用的 prev_stage_predictions")
    return result


def build_case_commands(
    use_best_config: bool,
    inference_parts_list: Optional[List[List[str]]],
    ensemble_parts: Optional[List[str]],
    task_name: str,
    model_type: str,
    trainer: str,
    plans: str,
    fold: int,
    folds: Optional[List[int]],
    input_folder: Path,
    output_folder: Path,
    prev_stage_runtime_path: Optional[str],
    continue_prediction: bool,
    num_processes_preprocessing: int,
    num_processes_saving: int,
) -> Tuple[List[List[str]], List[Path]]:
    """
    为当前输入目录组生成最终命令列表（predict / ensemble 等）。
    1. 先构建基础命令（占位符替换或自定义 -i -o）
    2. 再对每条 nnUNetv2_predict 补充运行期参数（-device、--continue_prediction、-npp、-nps 等）；
      nnUNetv2_ensemble 不修改。
    """
    if use_best_config and inference_parts_list is not None:
        base_cmds, extra_tmp_dirs = build_best_config_base_commands(
            inference_parts_list,
            ensemble_parts,
            input_folder,
            output_folder,
            prev_stage_runtime_path,
        )
        require_probabilities = ensemble_parts is not None
    else:
        base_cmds = [
            build_custom_base_command(
                task_name,
                model_type,
                trainer,
                plans,
                fold,
                folds,
                input_folder,
                output_folder,
            )
        ]
        extra_tmp_dirs = []
        require_probabilities = False

    final_cmds = [
        append_runtime_predict_options(
            cmd,
            prev_stage_runtime_path,
            continue_prediction,
            num_processes_preprocessing,
            num_processes_saving,
            require_probabilities=require_probabilities,
        )
        for cmd in base_cmds
    ]
    return final_cmds, extra_tmp_dirs