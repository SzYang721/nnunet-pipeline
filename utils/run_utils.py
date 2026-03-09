import time
from datetime import datetime

def monitor_output_folder(output_paths, stop_event, total_files):
    """
    监控期望的输出文件是否已生成，定期打印进度与 ETA。

    参数:
        output_paths: 期望的输出文件路径列表（每个病例一个路径，用于 p.exists() 统计）
        stop_event: 停止事件，主流程结束后 set() 以结束本线程
        total_files: 待处理总文件数（应与 len(output_paths) 一致）
    """
    # 初始化计数
    processed_count_last = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        # 计算当前已处理文件数
        processed_count = sum(1 for p in output_paths if p.exists())
        
        # 如果有新文件处理完成，显示进度
        if processed_count > processed_count_last:
            # 计算进度百分比
            progress = (processed_count / total_files) * 100
            
            # 计算时间信息
            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            
            # 计算预计剩余时间
            if processed_count > 0:
                time_per_file = elapsed / processed_count
                remaining_files = total_files - processed_count
                remaining_time = time_per_file * remaining_files
                remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                
                # 估计完成时间
                eta = datetime.now().timestamp() + remaining_time
                eta_str = datetime.fromtimestamp(eta).strftime("%H:%M:%S")
                
                # 计算处理速度
                newly_processed = processed_count - processed_count_last
                if newly_processed > 0:
                    time_since_last_update = time.time() - (start_time + (processed_count_last * time_per_file))
                    speed = newly_processed / time_since_last_update if time_since_last_update > 0 else 0
                    speed_str = f"{speed:.2f} 文件/秒"
                else:
                    speed_str = "计算中..."
            else:
                remaining_str = "计算中..."
                eta_str = "计算中..."
                speed_str = "计算中..."
            
            # 创建进度条
            bar_length = 30
            filled_length = int(bar_length * processed_count / total_files)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # 打印进度信息
            print(f"\n{'='*80}")
            print(f"🔄 预测进度: {processed_count}/{total_files} ({progress:.1f}%)")
            print(f"[{bar}]")
            print(f"  - 已处理: {processed_count} 个文件")
            print(f"  - 已用时间: {elapsed_str}")
            print(f"  - 预计剩余: {remaining_str}")
            print(f"  - 预计完成: {eta_str}")
            print(f"  - 处理速度: {speed_str}")
            print(f"{'='*80}\n")
            
            # 更新上次处理数量
            processed_count_last = processed_count
        
        # 等待一段时间再检查
        time.sleep(5)
