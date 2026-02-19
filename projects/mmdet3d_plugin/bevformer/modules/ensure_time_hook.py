from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class EnsureTimeDataHook(Hook):
    """Avoid KeyError: data_time/time in TextLoggerHook after eval."""

    def after_train_iter(self, runner):
        out = runner.log_buffer.output
        out.setdefault('time', 0.0)
        out.setdefault('data_time', 0.0)
