from .base import BaseShardingManager


class MegatronNaiveShardingManager(BaseShardingManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)