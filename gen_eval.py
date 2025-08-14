import json
import argparse
import os
import stat
import multiprocessing

def main():
    parser = argparse.ArgumentParser(description="根据JSON配置文件生成一个用于批量评估的Bash脚本。")
    parser.add_argument('--jobs_config', type=str, default='evaluation_jobs.json', help='包含所有评估任务的JSON配置文件路径。')
    parser.add_argument('--output_script', type=str, default='run_all_evaluations.sh', help='要生成的Bash脚本的文件名。')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(), help='为每个并行评估任务分配的进程数。')
    args = parser.parse_args()

    # 检查jobs配置文件是否存在
    if not os.path.exists(args.jobs_config):
        print(f"错误: 任务配置文件 '{args.jobs_config}' 未找到！")
        print("请先创建一个包含评估任务的JSON文件。")
        return

    # 读取JSON配置文件
    with open(args.jobs_config, 'r') as f:
        try:
            jobs_data = json.load(f)
            jobs = jobs_data.get("jobs", [])
        except json.JSONDecodeError:
            print(f"错误: 无法解析JSON文件 '{args.jobs_config}'。请检查其格式。")
            return

    if not jobs:
        print(f"在 '{args.jobs_config}' 中未找到任何作业。")
        return

    # 开始生成Bash脚本内容
    script_content = []
    script_content.append("#!/bin/bash")
    script_content.append("# 这是一个自动生成的脚本，用于按顺序执行多个并行评估任务。")
    script_content.append("# 使用 'bash run_all_evaluations.sh' 来运行它。")
    script_content.append("")
    script_content.append("set -e  # 如果任何命令失败，则立即退出脚本")
    script_content.append("")

    # 为每个作业生成一条命令
    for i, job in enumerate(jobs):
        base_dir = job.get("base_dir")
        config = job.get("config")
        gt_mesh = job.get("gt_mesh")

        if not all([base_dir, config, gt_mesh]):
            print(f"警告: 作业 {i+1} 缺少必要的键 ('base_dir', 'config', 'gt_mesh')，已跳过。")
            continue

        script_content.append(f"# --- 任务 {i+1}: 评估场景 {os.path.basename(base_dir)} ---")
        command = (
            f"python para_eval.py \\\n"
            f"    --base_dir {base_dir} \\\n"
            f"    --config {config} \\\n"
            f"    --gt_mesh {gt_mesh} \\\n"
            f"    --workers {args.workers}"
        )
        script_content.append(command)
        script_content.append("")

    # 将内容写入输出文件
    try:
        with open(args.output_script, 'w') as f:
            f.write("\n".join(script_content))
        
        # 使生成的脚本文件可执行
        st = os.stat(args.output_script)
        os.chmod(args.output_script, st.st_mode | stat.S_IEXEC)

        print(f"成功生成Bash脚本: '{args.output_script}'")
        print(f"现在，您可以通过在终端中运行以下命令来执行所有评估任务:")
        print(f"bash {args.output_script}")

    except IOError as e:
        print(f"错误: 无法写入脚本文件 '{args.output_script}': {e}")


if __name__ == '__main__':
    main()