import glob
import shutil
import os
import os.path as osp
import json
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.data import BaseData
from torch_geometric.datasets import AirfRANS
from torch_geometric.io import read_off


class HydroDataset(InMemoryDataset):
    url = "https://github.com/abelinsky/datasets/raw/main/hydropipelines.zip"

    def __init__(
        self,
        root: str | None = None,
        train: bool = True,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        pre_filter: Callable[..., Any] | None = None,
        log: bool = True,
    ):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> str | List[str] | Tuple:
        return "net1.json"

    @property
    def processed_file_names(self) -> str | List[str] | Tuple:
        return "data.pt"

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root, log=True)
        os.rename(osp.join(self.root, "hydro_pipelines"), self.raw_dir)
        os.unlink(path)

    def process(self):
        path = osp.join(self.raw_dir, "*.json")
        pipenets = glob.glob(path)

        data_list = []
        for net in pipenets:
            with open(net) as fjson:
                parsed = json.load(fjson)

            nodes = parsed["nodes"]
            nodes_info = [
                [node["attributes"]["q"], node["attributes"]["p"]] for node in nodes
            ]
            x = torch.tensor(nodes_info, dtype=torch.float)

            edges = parsed["links"]
            edge_indices = [[edge["source"], edge["target"]] for edge in edges]
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

            edge_attr = torch.tensor(
                [
                    [edge["attributes"]["length"], edge["attributes"]["diam_in"]]
                    for edge in edges
                ],
                dtype=torch.float32,
            )

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
