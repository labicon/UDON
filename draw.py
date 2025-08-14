import os
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd

def smooth_data(data, window_size):
    """使用移动平均对数据进行平滑处理。"""
    if window_size <= 1 or len(data) < window_size:
        return data
    # 使用 pandas 的 rolling window 功能进行平滑，更稳健
    # center=True 使窗口居中, min_periods=1 处理边缘数据
    series = pd.Series(data)
    smoothed_series = series.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed_series.to_numpy()

def extract_scalar_data(log_path, tag_name):
    """从单个 TensorBoard 日志文件中提取指定 tag 的标量数据。"""
    try:
        ea = event_accumulator.EventAccumulator(
            log_path,
            size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()

        if tag_name not in ea.Tags()['scalars']:
            print(f"警告: 在文件 {log_path} 中未找到 Tag '{tag_name}'。")
            return None, None

        events = ea.Scalars(tag_name)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        
        return steps, values

    except Exception as e:
        print(f"错误: 处理文件 {log_path} 时发生错误: {e}")
        return None, None

def aggregate_and_process_data(data_per_agent):
    """对齐、插值并计算多个 agent 数据的均值和标准差。"""
    if not data_per_agent:
        return None, None, None

    min_step = min(d[0][0] for d in data_per_agent if len(d[0]) > 0)
    max_step = max(d[0][-1] for d in data_per_agent if len(d[0]) > 0)
    common_steps = np.linspace(min_step, max_step, 500)

    interpolated_values = []
    for steps, values in data_per_agent:
        if len(steps) > 1:
            interp_vals = np.interp(common_steps, steps, values)
            interpolated_values.append(interp_vals)

    if not interpolated_values:
        return None, None, None

    values_matrix = np.vstack(interpolated_values)
    mean_values = np.mean(values_matrix, axis=0)
    std_values = np.std(values_matrix, axis=0)
    
    return common_steps, mean_values, std_values

def main():
    parser = argparse.ArgumentParser(description="以RL风格将多种损失绘制在不同的子图中。")
    parser.add_argument('--base_dir', required=True, type=str, 
                        help='包含所有 agent 文件夹的实验根目录 (例如: RAMEN2_scene0106_D99)')
    parser.add_argument('--output_file', type=str, default='rl_style_subplots.png', 
                        help='输出图像的文件名')
    parser.add_argument('--title', type=str, default=None,
                        help='图表的总标题 (默认为实验目录名)')
    parser.add_argument('--smoothing_window', type=int, default=20,
                        help='移动平均的窗口大小。设为1则不进行平滑。')

    args = parser.parse_args()

    tags_to_plot = ['Uncertainty/Mean', 'Uncertainty/Max']
    

    
    agent_dirs_pattern = os.path.join(args.base_dir, 'agent_*')
    agent_dirs = sorted(glob.glob(agent_dirs_pattern))

    if not agent_dirs:
        print(f"错误: 在 '{args.base_dir}' 目录下未找到任何 'agent_*' 文件夹。")
        return

    print(f"在 '{args.base_dir}' 中找到 {len(agent_dirs)} 个 agent。")

    data_by_tag = {tag: [] for tag in tags_to_plot}
    for agent_dir in agent_dirs:
        log_file_pattern = os.path.join(agent_dir, 'logs', 'events.out.tfevents.*')
        log_files = glob.glob(log_file_pattern)
        if not log_files: continue
        
        log_path = log_files[0]
        for tag in tags_to_plot:
            steps, values = extract_scalar_data(log_path, tag)
            if steps is not None and values is not None:
                data_by_tag[tag].append((steps, values))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    for i, tag in enumerate(tags_to_plot):
        ax = axes[i]
        print(f"正在处理 Tag: {tag}...")
        data_for_this_tag = data_by_tag[tag]
        
        if len(data_for_this_tag) < 2:
            print(f"  -> 警告: Tag '{tag}' 的有效数据源少于2个，无法计算标准差，已跳过。")
            ax.text(0.5, 0.5, f"Data for '{tag}'\nis insufficient", ha='center', va='center', transform=ax.transAxes)
            continue

        common_steps, mean_values, std_values = aggregate_and_process_data(data_for_this_tag)
        
        if common_steps is None:
            print(f"  -> 警告: 无法处理 Tag '{tag}' 的数据。")
            continue

        # 对均值和标准差进行平滑处理
        smoothed_mean = smooth_data(mean_values, args.smoothing_window)
        smoothed_std = smooth_data(std_values, args.smoothing_window)

        # 绘制
        ax.plot(common_steps, smoothed_mean, color=f'C{i}', label=f'Mean {tag}')
        ax.fill_between(common_steps, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, 
                        color=f'C{i}', alpha=0.2, label=f'Std Dev {tag}')
        
        ax.set_title(tag, fontsize=14)
        ax.grid(True)
        ax.set_ylabel("Loss Value")

    for ax in axes[2:]:
        ax.set_xlabel("Step", fontsize=12)

    main_title = args.title if args.title else f"Agent Performance for {os.path.basename(args.base_dir)}"
    fig.suptitle(main_title, fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.output_file, dpi=300)
    print(f"\n图表已成功保存到: {args.output_file}")

if __name__ == '__main__':
    main()