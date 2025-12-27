# -*- coding: utf-8 -*-
"""
TDMPC2 Package Initialization
"""

# 导入核心模块
from . import common
from . import envs
from . import trainer

# 版本信息
__version__ = "1.0.0"
__author__ = "TDMPC2 Team"

# 导出主要组件
__all__ = [
    "common",
    "envs", 
    "trainer"
]