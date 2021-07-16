from typing import Dict, Any, Iterable, Callable


class Node(object):
    def __init__(self):
        pass

    def get_dataloader(self, cfg: Dict[str, Any], preprocess: Callable) -> Iterable:
        pass

    def load_state(self, task):
        pass

    def dump_state(self, task):
        pass

    def merge_weight(self, task):
        pass
