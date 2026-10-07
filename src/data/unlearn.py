from collections.abc import Mapping

import torch
from torch.utils.data import Dataset


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, anchor="forget"):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
        """
        self.forget = forget
        self.retain = retain
        self.anchor = anchor

    @staticmethod
    def _split_len(split):
        if isinstance(split, Mapping):
            lengths = [len(dataset) for dataset in split.values()]
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"All datasets in a split must have the same length: {lengths}"
                )
            return lengths[0]
        return len(split)

    @staticmethod
    def _split_get(split, idx):
        if isinstance(split, Mapping):
            return {name: dataset[idx] for name, dataset in split.items()}
        return split[idx]

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.anchor == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when anchor=forget"
            )
            return self._split_len(self.forget)
        elif self.anchor == "retain":
            assert self.retain is not None, ValueError(
                "retain dataset can't be None when anchor=retain"
            )
            return self._split_len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            item["forget"] = self._split_get(self.forget, idx)
            if self.retain:
                retain_idx = torch.randint(0, self._split_len(self.retain), (1,)).item()
                item["retain"] = self._split_get(self.retain, retain_idx)
        elif self.anchor == "retain":
            item["retain"] = self._split_get(self.retain, idx)
            if self.forget:
                forget_idx = torch.randint(0, self._split_len(self.forget), (1,)).item()
                item["forget"] = self._split_get(self.forget, forget_idx)
        return item
