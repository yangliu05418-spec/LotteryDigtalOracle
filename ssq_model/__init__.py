"""双色球概率建模研究工具包。

本包只用于组合数学、概率分布、统计检验与模型回测研究；
不提供现实投注建议，也不承诺任何命中能力。
"""

from .data import Draw, load_draws, validate_draws

__all__ = ["Draw", "load_draws", "validate_draws"]
__version__ = "0.1.0"
