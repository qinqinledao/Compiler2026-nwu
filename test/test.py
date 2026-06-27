#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Triton Kernel Tester
用于单发快速测试 test.txt 中的算子代码，跳过繁琐的 EA 演进流程。
"""

import os
import sys
import json
import argparse
from pathlib import Path

# ================= 核心：跨级目录动态挂载 =================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
# =======================================================

from config import EAConfig
from executor import TritonExecutor
from optimizer_agent import get_baseline_from_json


def find_test_file(kernel_path: Path, kernel_name: str) -> Path:
    """查找测试文件，优先级：test_1 > test_2 > test_3 > test"""
    for test_case_id in range(1, 4):
        test_file = kernel_path / f"test_{kernel_name}_{test_case_id}.py"
        if test_file.exists():
            print(f"[Loader] 找到测试文件: {test_file.name}")
            return test_file, test_case_id
    test_file = kernel_path / f"test_{kernel_name}.py"
    if test_file.exists():
        print(f"[Loader] 找到测试文件: {test_file.name}")
        return test_file, 1
    raise ValueError(f"找不到 {kernel_name} 的测试文件")


def main():
    parser = argparse.ArgumentParser(description="Standalone Triton Kernel Tester")

    parser.add_argument("--code", type=str, default=str(CURRENT_DIR / "test.txt"),
                        help="存放算子源码的文件 (默认: test/test.txt)")
    parser.add_argument("--kernel", type=str, default="_rms_norm_kernel",
                        help="待测试的算子名称")
    parser.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "datasets"),
                        help="数据集目录 (默认: datasets/)")
    parser.add_argument("--baseline", type=str,
                        default=str(PROJECT_ROOT / "baseline" / "baseline.json"),
                        help="baseline.json 路径")
    parser.add_argument("--work-dir", type=str, default=str(PROJECT_ROOT / "test"),
                        help="工作目录，performance 数据会存在这里 (默认: test/)")
    args = parser.parse_args()

    code_path = Path(args.code)
    if not code_path.exists():
        print(f"[!] 找不到代码文件: {code_path}")
        print(f"请在 {CURRENT_DIR}/ 目录下创建 test.txt 并贴入算子代码。")
        return

    with open(code_path, 'r', encoding='utf-8') as f:
        code = f.read()

    if not code.strip():
        print(f"[!] {code_path} 是空的！")
        return

    dataset_dir = Path(args.dataset)
    kernel_dir = dataset_dir / args.kernel
    if not kernel_dir.exists():
        print(f"[!] 找不到算子目录: {kernel_dir}")
        return

    print(f"{'=' * 60}")
    print(f"🚀 [Standalone Tester] 正在测试算子: {args.kernel}")
    print(f"📁 代码来源: {code_path}")
    print(f"📊 数据集: {dataset_dir}")
    print(f"{'=' * 60}")

    try:
        test_file, test_case_id = find_test_file(kernel_dir, args.kernel)
        baseline_time = get_baseline_from_json(args.baseline, args.kernel, test_case_id)
        if baseline_time is None:
            print(f"[!] 初始化失败: baseline.json 中找不到 {args.kernel} 的 test_case_{test_case_id} 数据")
            return
    except (ValueError, FileNotFoundError) as e:
        print(f"[!] 初始化失败: {e}")
        return

    config = EAConfig()

    try:
        executor = TritonExecutor(
            baseline_time=baseline_time,
            test_code_path=str(test_file),
            config=config,
            kernel_name=args.kernel,
            work_dir=Path(args.work_dir)
        )
    except Exception as e:
        print(f"[!] Executor 初始化失败: {e}")
        return

    print(f"[*] 基线时间: {baseline_time:.2f} us")
    print(f"[*] 开始测评")

    result = executor.evaluate(code)

    print(f"\n{'=' * 60}")
    print(f"🎯 测试完成!")
    print(f"{'=' * 60}")

    if result.success:
        print(f"✅ 状态: 运行成功！")
        print(f"⏱️  运行时间 (Latency): {result.execution_time:.2f} us")
        print(f"📉 基线时间 (Baseline): {executor.baseline_time:.2f} us")
        print(f"🚀 实际加速比 (Speedup): {result.speedup:.4f}x")
        print(f"💯 适应度得分 (Fitness): {result.fitness:.4f}")
    else:
        print(f"❌ 状态: 真机运行或编译失败！")
        print(f"⚠️  错误信息分析片段:")
        real_error = result.error
        try:
            log_path = Path(executor.performance_dir) / executor.kernel_name / "get_prof.log"
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read().strip()
                    if log_content:
                        real_error = log_content[-2000:]
        except Exception:
            pass
        print(real_error if real_error else "未知错误，请检查算子代码语法是否正确。")


if __name__ == "__main__":
    main()
