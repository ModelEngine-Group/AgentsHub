"""
operators/base.py
================
算子基类定义。

所有算子（Operator）都应继承 BaseOperator，
并实现 run() 方法。
"""


class OperatorRuntimeError(Exception):
    """算子运行时异常"""
    pass


class BaseOperator:
    """
    算子基类。

    所有算子继承此类，实现以下属性和方法：

    Attributes:
        name: 算子名称
        description: 算子功能描述
        input_schema: 输入数据格式描述（dict）
        output_schema: 输出数据格式描述（dict）
    """

    name: str = "BaseOperator"
    description: str = "算子基类"
    input_schema: dict = {}
    output_schema: dict = {}

    def __init__(self):
        self.status = "pending"  # pending / running / success / failed
        self.error_message = None
        self.result = None

    def run(self, *args, **kwargs):
        """
        执行算子逻辑。

        子类必须重写此方法。

        Args:
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            算子执行结果

        Raises:
            OperatorRuntimeError: 执行失败时抛出
        """
        raise NotImplementedError("子类必须实现 run() 方法")

    def execute(self, *args, **kwargs):
        """
        执行入口，带异常捕获和状态追踪。

        推荐外部调用此方法，而非直接调用 run()。
        """
        self.status = "running"
        try:
            self.result = self.run(*args, **kwargs)
            self.status = "success"
            return self.result
        except Exception as e:
            self.status = "failed"
            self.error_message = str(e)
            raise OperatorRuntimeError(f"[{self.name}] 执行失败: {e}") from e

    def __repr__(self):
        return f"<{self.__class__.__name__} status={self.status}>"
