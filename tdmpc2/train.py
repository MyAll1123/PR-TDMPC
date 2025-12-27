#!/usr/bin/env python3
"""
训练脚本 - 调用 tdmpc2 模块的训练功能
"""

import sys
import os

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # 导入并运行 tdmpc2 模块的训练函数
    from tdmpc2.train import train
    train()