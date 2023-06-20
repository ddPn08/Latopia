from typing import Any


class BasePipeline:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
