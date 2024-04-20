import torch
import kornia
from typing import Tuple

class Clahe(torch.nn.Module):

    def __init__(self, clip_limit: int = 40, grid_size: Tuple[int, int] = (8, 8)) -> None:
        super().__init__()
        self.clip_limit, self.grid_size = float(clip_limit), grid_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return kornia.enhance.equalize_clahe(img, self.clip_limit, self.grid_size)

    def __repr__(self) -> str:
        return "{}(clip_limit={}, tile_grid_size={})".format(
            self.__class__.__name__,
            self.clip_limit,
            self.grid_size
        )
