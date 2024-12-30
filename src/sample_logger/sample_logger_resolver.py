from .base_sample_logger import BaseSampleLogger
from .SampleLoggerPass import SampleLoggerPass
from .SampleLoggerWithoutTarget import SampleLoggerWithoutTarget
from .SampleLoggerWithTarget import SampleLoggerWithTarget

def get_sample_logger(name : str, **kwargs) -> BaseSampleLogger:
    if name == "pass":
        return SampleLoggerPass()
    if name == "with_target":
        return SampleLoggerWithTarget(**kwargs)
    elif name == "without_target":
        return SampleLoggerWithoutTarget(**kwargs)
    else:
        raise NotImplementedError(f"cfg_sample_logger {name} not implemented")