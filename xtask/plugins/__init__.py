# xTask 自定义 evalscope 插件包。
#
# 此包仅作为命名空间使用：真正的插件模块位于
# ``xtask.plugins.evalscope`` 之下，由 wrapper 入口
# ``xtask.plugins.evalscope.run_gedit_perf`` 在调用
# ``run_perf_benchmark`` 之前显式 import，以触发
# ``@register_api`` 装饰器完成注册（evalscope 的
# ``ApiRegistry`` 只在插件模块被 import 时注册，不做外部扫描）。
