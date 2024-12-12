from pytorch_lightning.callbacks import TQDMProgressBar
class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        # Remove "v_num" and other metrics like speed from the bar
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)  # Remove version number
        return items
