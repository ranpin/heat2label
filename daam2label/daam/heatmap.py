import cv2
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Dict, Tuple, Set, Iterable

from matplotlib import pyplot as plt
import matplotlib.patches as patches

import numpy as np
import PIL.Image
import spacy.tokens
import torch
import torch.nn.functional as F

from .evaluate import compute_ioa
from .utils import compute_token_merge_indices, cached_nlp, auto_autocast

__all__ = ['GlobalHeatMap', 'RawHeatMapCollection', 'WordHeatMap', 'ParsedHeatMap', 'SyntacticHeatMapPair']


def plot_overlay_heat_map(im, heat_map, src_size, word=None, out_file=None, crop=None, color_normalize=True, ax=None):
    
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, plt.Axes) -> None
    if ax is None:
        plt.clf()
        plt.rcParams.update({'font.size': 24})
        plt_ = plt
    else:
        plt_ = ax

    # im = im.resize(src_size) # To get into right size

    with auto_autocast(dtype=torch.float32):
        im = np.array(im)

        if crop is not None:
            heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
            im = im[crop:-crop, crop:-crop]

        if color_normalize:
            heat_map = heat_map.squeeze().cpu().numpy()
            # heat_map = cv2.resize(heat_map, src_size, interpolation=cv2.INTER_CUBIC)  # to expand
            bbox_list = crop_heat_map_bbox(heat_map)
            visualize_tool(im, bbox_list, [word for _ in range(len(bbox_list))], heat_map)


def return_overlay_heat_map(im, heat_map, src_size, word=None, out_file=None, crop=None, color_normalize=True, ax=None):
    
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, plt.Axes) -> None
    if ax is None:
        plt.clf()
        plt.rcParams.update({'font.size': 24})
        plt_ = plt
    else:
        plt_ = ax

    im = im.resize(src_size) # To get into right size

    with auto_autocast(dtype=torch.float32):
        im = np.array(im)

        if crop is not None:
            heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
            im = im[crop:-crop, crop:-crop]

        if color_normalize:
            heat_map = heat_map.squeeze().cpu().numpy()
            heat_map = cv2.resize(heat_map, src_size, interpolation=cv2.INTER_CUBIC)  # to expand
            bbox_list = crop_heat_map_bbox(heat_map)

            return im, heat_map, bbox_list


def plot_overlay_heat_map_old(im, heat_map, word=None, out_file=None, crop=None, color_normalize=True, ax=None):
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, plt.Axes) -> None
    if ax is None:
        plt.clf()
        plt.rcParams.update({'font.size': 24})
        plt_ = plt
    else:
        plt_ = ax

    with auto_autocast(dtype=torch.float32):
        im = np.array(im)

        if crop is not None:
            heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
            im = im[crop:-crop, crop:-crop]

        if color_normalize:
            plt_.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet')
        else:
            heat_map = heat_map.clamp_(min=0, max=1)
            plt_.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet', vmin=0.0, vmax=1.0)

        im = torch.from_numpy(im).float() / 255
        im = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1)
        plt_.imshow(im)

        if word is not None:
            if ax is None:
                plt.title(word)
            else:
                ax.set_title(word)

        if out_file is not None:
            plt.savefig(out_file)

def crop_heat_map_bbox(heat_map, threshold=0.5, min_size=1024):
    """
    heat_map -> ndarray: source heatmap
    threshold -> float: 0.3 is a good threshold, adaptable
    min_size -> int: minimized box size

    """
    from collections import deque
    from copy import deepcopy

    def bfs_search(i, j, thr, map):

        island = []
        queue = deque([(i, j)])
        while(queue):

            cur_i, cur_j = queue.popleft()
            if cur_i < 0 or cur_i >= map.shape[0] or \
                cur_j < 0 or cur_j >= map.shape[1] or \
                map[cur_i, cur_j] < thr:
                
                continue

            map[cur_i, cur_j] = 0.0
            island.append([cur_i, cur_j])

            queue.append((cur_i, cur_j + 1))
            queue.append((cur_i, cur_j - 1))
            queue.append((cur_i + 1, cur_j))
            queue.append((cur_i - 1, cur_j))

        return island

    heat_map = deepcopy(heat_map)

    max_val = heat_map.max()
    min_val = heat_map.min()

    count = 0
    island_list = []
    for i in range(heat_map.shape[0]):
        for j in range(heat_map.shape[1]):
            if heat_map[i, j] >= threshold:
                island = bfs_search(i, j, threshold, heat_map)
                island_list.append(np.array(island))
                count += 1

    bbox_list = []
    max_area = 0
    max_bbox = []
    for island in island_list:
        x1 = island[:,1].min()
        y1 = island[:,0].min()
        x2 = island[:,1].max()
        y2 = island[:,0].max()

        if abs(x2 - x1) * abs(y2 - y1) > max_area:
            max_area = abs(x2 - x1) * abs(y2 - y1)
            max_bbox = [[x1, y1, x2, y2]]

        if abs(x2 - x1) * abs(y2 - y1) <= min_size:
            continue

        bbox_list.append([
            x1, y1, x2, y2
        ]) # x1, y1, x2, y2

    
    return bbox_list
    # return max_bbox


def visualize_tool(img, bbox_list, label_list, heat_map, save_path=""):

    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    
    plt.imshow(img)
    plt.savefig("output/attentions/raw.png")
    plt.clf()

    plt.imshow(heat_map, cmap="jet")
    plt.savefig("output/attentions/heatmap.png")
    
    # put the heat_map to img
    heat_map = heat_map[:, :, np.newaxis]
    img = img / 255.0
    img = np.concatenate((img, (1 - heat_map)), axis=-1)

    assert len(bbox_list) == len(label_list), "Number of bboxes should equal to labels"

    for bbox, label in zip(bbox_list, label_list):
        # img = cv2.rectangle(
        #     img, 
        #     (bbox[0], bbox[1]), # x1, y1
        #     (bbox[2], bbox[3]), # x2, y2
        #     (255, 255, 0),
        #     2,
        #     # -1,
        # )
        # img = cv2.putText(
        #     img,
        #     label,
        #     (bbox[0], bbox[1]),
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     1, 
        #     (255, 255, 0), 
        #     1
        # )

        rec = patches.Rectangle(
            (bbox[0], bbox[1]), 
            bbox[2] - bbox[0], 
            bbox[3] - bbox[1], 
            fill=False, 
            color="y", 
            label="dog",
            linewidth=2,
        )
        plt.gca().add_patch(rec)
    
    plt.imshow(img)
    plt.savefig("output/attentions/bbox.png")

    # if save_path == "": save_path = "img_rec.png"
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path, img)


class WordHeatMap:
    def __init__(self, heatmap: torch.Tensor, word: str = None, word_idx: int = None):
        self.word = word
        self.word_idx = word_idx
        self.heatmap = heatmap

    @property
    def value(self):
        return self.heatmap

    def plot_overlay(self, image, src_size, out_file=None, color_normalize=True, ax=None, **expand_kwargs):
        # type: (PIL.Image.Image | np.ndarray, Path, bool, plt.Axes, Dict[str, Any]) -> None
        # plot_overlay_heat_map(
        #     image,
        #     self.expand_as(image, **expand_kwargs),
        #     src_size,
        #     word=self.word,
        #     out_file=out_file,
        #     color_normalize=color_normalize,
        #     ax=ax
        # )

        return return_overlay_heat_map(
            image,
            self.expand_as(image, **expand_kwargs),
            src_size,
            word=self.word,
            out_file=out_file,
            color_normalize=color_normalize,
            ax=ax
        )

    def expand_as(self, image, absolute=False, threshold=None, plot=False, **plot_kwargs):
        # type: (PIL.Image.Image, bool, float, bool, Dict[str, Any]) -> torch.Tensor
        im = self.heatmap.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(image.size[0], image.size[1]), mode='bicubic')

        if not absolute:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)

        if threshold:
            im = (im > threshold).float()

        im = im.cpu().detach().squeeze()

        if plot:
            self.plot_overlay(image, **plot_kwargs)

        return im

    def compute_ioa(self, other: 'WordHeatMap'):
        return compute_ioa(self.heatmap, other.heatmap)


@dataclass
class SyntacticHeatMapPair:
    head_heat_map: WordHeatMap
    dep_heat_map: WordHeatMap
    head_text: str
    dep_text: str
    relation: str


@dataclass
class ParsedHeatMap:
    word_heat_map: WordHeatMap
    token: spacy.tokens.Token


class GlobalHeatMap:
    def __init__(self, tokenizer: Any, prompt: str, heat_maps: torch.Tensor):
        self.tokenizer = tokenizer
        self.heat_maps = heat_maps
        self.prompt = prompt
        self.compute_word_heat_map = lru_cache(maxsize=50)(self.compute_word_heat_map)

    def compute_word_heat_map(self, word: str, word_idx: int = None, offset_idx: int = 0) -> WordHeatMap:
        merge_idxs, word_idx = compute_token_merge_indices(self.tokenizer, self.prompt, word, word_idx, offset_idx)
        return WordHeatMap(self.heat_maps[merge_idxs].mean(0), word, word_idx)

    def parsed_heat_maps(self) -> Iterable[ParsedHeatMap]:
        for token in cached_nlp(self.prompt):
            try:
                heat_map = self.compute_word_heat_map(token.text)
                yield ParsedHeatMap(heat_map, token)
            except ValueError:
                pass

    def dependency_relations(self) -> Iterable[SyntacticHeatMapPair]:
        for token in cached_nlp(self.prompt):
            if token.dep_ != 'ROOT':
                try:
                    dep_heat_map = self.compute_word_heat_map(token.text)
                    head_heat_map = self.compute_word_heat_map(token.head.text)

                    yield SyntacticHeatMapPair(head_heat_map, dep_heat_map, token.head.text, token.text, token.dep_)
                except ValueError:
                    pass


RawHeatMapKey = Tuple[int, int, int]  # factor, layer, head


class RawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(lambda: 0.0)
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)

    def update(self, factor: int, layer_idx: int, head_idx: int, heatmap: torch.Tensor):
        with auto_autocast(dtype=torch.float32):
            key = (factor, layer_idx, head_idx)
            self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def factors(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
