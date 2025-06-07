#!/usr/bin/env python3
"""
启用形态学后处理的DGCNN推理脚本
"""

import subprocess
import sys

def main():
    """运行带形态学后处理的DGCNN推理"""
    
    cmd = [
        sys.executable, "pytorch/main.py",
        "--model_path", "pytorch/checkpoints/verify_three_classifications_with_matlab_data/models/model.t7",
        "--predict", "True",
        "--exp_name", "verify_three_classifications_with_matlab_data",
        "--predict_path", "pytorch/dataset/test_set",
        
        # 启用形态学后处理
        "--use_morphological_postprocess", "True",
        "--morph_k_neighbors", "6",        # 邻居数量
        "--morph_min_component_size", "8", # 最小组件大小
        "--morph_max_hole_size", "4"       # 最大空洞大小
    ]
    
    print("启动带形态学后处理的DGCNN推理...")
    print("后处理配置:")
    print(f"  - k邻居数: 6")
    print(f"  - 预期时间增加: +0.5-1.0秒/文件")
    print(f"  - 预期标签改变率: 3-5%")
    print()
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n推理完成! 退出代码: {result.returncode}")
        print("结果已保存到 pytorch/result/ 目录下")
        
    except subprocess.CalledProcessError as e:
        print(f"执行失败! 退出代码: {e.returncode}")
        print("请检查错误信息并修正问题")
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
        
    except Exception as e:
        print(f"执行时发生错误: {e}")

if __name__ == "__main__":
    main() 