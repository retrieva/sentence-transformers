from typing import Callable, List


def collate_fn(functions: List[Callable]) -> Callable:
    def collate_fn_(batch):
        for function in functions:
            batch = function(batch)
        return batch

    return collate_fn_
