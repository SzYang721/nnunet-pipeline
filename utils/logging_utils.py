import logging
from pathlib import Path
import sys

# 日志目录：仓库根目录下的 log（与 train_with_nnUNet / predict_with_nnUNet 共用）
LOGS_DIR = Path(__file__).resolve().parent.parent / "log"

def setup_logging(log_name=None, log_dir=None, level=logging.INFO, overwrite=False):
    """
    统一日志配置。默认使用 LOGS_DIR，文件名为当前脚本名.log。
    参数：
        log_name: 日志文件名（不含路径），如 None 则用调用者脚本名
        log_dir: 日志目录，默认使用 LOGS_DIR
        level: 日志等级
        overwrite: 是否覆盖日志文件（True=覆盖，False=追加）
    """
    import __main__
    if log_dir is None:
        log_dir = LOGS_DIR
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    if log_name is None:
        log_name = f"{Path(__main__.__file__).stem}.log"
    log_path = log_dir / log_name
    handlers = [
        logging.FileHandler(log_path, mode='w' if overwrite else 'a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logging.info(f"日志已初始化: {log_path}")
    return log_path