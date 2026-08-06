"""Provider-free harness experiment helpers.

实验模块只消费 FirstCoder 已生成的 run artifacts；它们不拥有 agent runtime，
也不应把 benchmark 过程中的数据写回 memory store。
"""
