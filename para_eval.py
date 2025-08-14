import os
import glob
import argparse
import subprocess
import multiprocessing
from tqdm import tqdm
import re  # 导入正则表达式模块

def extract_number_from_path(filepath):
    """从文件路径中提取数字用于自然排序，例如从 'mesh_track773.ply' 提取 773。"""
    filename = os.path.basename(filepath)
    # 使用正则表达式查找文件名中的数字部分
    match = re.search(r'(\d+)\.ply$', filename)
    if match:
        return int(match.group(1))
    return 0  # 如果没有找到数字，则返回0作为默认值

def run_evaluation_task(job):
    """
    执行单个评估任务：先运行 cull_mesh.py，然后运行 eval_recon.py。
    这个函数将在一个独立的进程中被调用。
    """
    input_mesh, config_path, gt_mesh_path = job
    task_name = os.path.relpath(input_mesh, os.path.dirname(config_path))
    
    try:
        # --- 步骤 1: 运行 cull_mesh.py ---
        print(f"[{task_name}] 开始剔除网格 (culling)...")
        cull_command = [
            'python', 'cull_mesh.py',
            '--config', config_path,
            '--input_mesh', input_mesh,
            '--remove_occlusion', '--gt_pose'
        ]
        subprocess.run(cull_command, check=True, capture_output=True, text=True)
        
        # --- 步骤 2: 运行 eval_recon.py ---
        # 确定 cull_mesh.py 的输出文件名
        culled_mesh_path = input_mesh.replace('.ply', '_cull_occlusion.ply')
        
        if not os.path.exists(culled_mesh_path):
            print(f"[{task_name}] 错误: 剔除后的网格文件 {culled_mesh_path} 未找到！")
            return (task_name, "失败: 剔除文件未生成")

        print(f"[{task_name}] 开始评估重建 (evaluating)...")
        eval_command = [
            'python', 'eval_recon.py',
            '--rec_mesh', culled_mesh_path,
            '--gt_mesh', gt_mesh_path,
            '-3d'
        ]
        # 运行并捕获输出
        result = subprocess.run(eval_command, check=True, capture_output=True, text=True)
        
        print(f"[{task_name}] 评估完成。")
        # 返回结果以便后续处理
        return (task_name, result.stdout)

    except subprocess.CalledProcessError as e:
        # 如果任何一个命令失败，则捕获错误
        print(f"[{task_name}] 任务失败！")
        print(f"命令: {' '.join(e.cmd)}")
        print(f"错误输出:\n{e.stderr}")
        return (task_name, f"失败:\n{e.stderr}")
    except Exception as e:
        print(f"[{task_name}] 发生未知错误: {e}")
        return (task_name, f"失败: {e}")

# ... (import 语句和 run_evaluation_task 函数保持不变) ...

def main():
    parser = argparse.ArgumentParser(description="并行执行多个SLAM实验的评估。")
    parser.add_argument('--base_dir', required=True, type=str, help='包含所有实验结果的根目录 (例如: output/scannet/scene0106_00)')
    parser.add_argument('--config', required=True, type=str, help='用于评估的配置文件路径 (例如: configs/scannet/scene0106.yaml)')
    parser.add_argument('--gt_mesh', required=True, type=str, help='作为基准的真值网格文件路径')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help='要使用的并行进程数 (默认为CPU核心数)')
    args = parser.parse_args()

    # --- 核心修改：自动发现并筛选出每个 agent 的最后一个网格文件 ---
    print("正在搜索需要评估的网格文件...")
    # 1. 找到所有 agent 文件夹
    agent_dirs_pattern = os.path.join(args.base_dir, 'agent_*')
    agent_dirs = glob.glob(agent_dirs_pattern)
    print(args.base_dir)
    final_meshes_to_evaluate = []
    for agent_dir in agent_dirs:
        # 2. 在每个 agent 文件夹内，找到所有 mesh_track*.ply 文件
        mesh_files_pattern = os.path.join(agent_dir, 'mesh_track*.ply')
        mesh_files = glob.glob(mesh_files_pattern)
        
        # 排除已经被剔除处理过的文件
        mesh_files = [m for m in mesh_files if '_cull' not in m]

        if not mesh_files:
            continue

        # 3. 按文件名中的数字进行自然数排序，并选择最后一个
        mesh_files.sort(key=extract_number_from_path)
        final_mesh = mesh_files[-1]
        final_meshes_to_evaluate.append(final_mesh)
        print(f"  -> 找到待评估文件: {os.path.relpath(final_mesh, args.base_dir)}")

    if not final_meshes_to_evaluate:
        print(f"在 '{args.base_dir}' 中未找到任何需要评估的最终网格文件。请检查路径。")
        return

    print(f"\n总共发现 {len(final_meshes_to_evaluate)} 个最终评估作业。")
    jobs = [(mesh_path, args.config, args.gt_mesh) for mesh_path in final_meshes_to_evaluate]

    # --- 使用进程池并行执行所有作业 ---
    print(f"使用 {args.workers} 个并行进程开始评估...")
    with multiprocessing.Pool(processes=args.workers) as pool:
        # 使用 tqdm 显示进度条
        results = list(tqdm(pool.imap_unordered(run_evaluation_task, jobs), total=len(jobs)))

    print("\n--- 所有评估任务已完成 ---")
    # (可选) 在这里可以添加代码来解析和汇总 `results` 列表中的所有评估结果
    for task, output in sorted(results):
        print(f"\n--- 结果来自: {task} ---")
        print(output)


if __name__ == '__main__':
    main()