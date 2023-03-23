import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
import cv2

from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.extras()` before the task is started
    - Calling the `utils.close_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def wrap(cfg: DictConfig):

        # apply extra utilities
        extras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")  # save exception to `.log` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.log")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            save_file(path, content)  # save task execution time (even if exception occurs)
            close_loggers()  # close loggers (even if exception occurs so multirun won't fail)

        log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


"""ELabel"""

import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np

def elabel_log_img(model, data, target, output):
    image, label_box, origin_image = data[0].to('cpu'), data[1].to('cpu'), data[2].to('cpu')
    batch_size = image.shape[0]
    # process boxes
    boxes, labels = [], []
    for i in range(batch_size):
        w, h = image[i].shape[1], image[i].shape[2]
        box = [label_box[i][0], label_box[i][1], label_box[i][0]+label_box[i][2], label_box[i][1]+label_box[i][3]]
        box = torch.tensor([box[0]*w, box[1]*h, box[2]*w, box[3]*h]).type(torch.int32)
        boxes.append([box])
        labels.append([str(target[i])])
    # draw boxes
    imgs_with_boxes = []
    for i in range(batch_size):
        img_res = draw_bounding_boxes((image[i]*256).type(torch.uint8), torch.stack(boxes[i]))
        imgs_with_boxes.append((origin_image[i]*256).type(torch.uint8)) # TODO: draw box on origin image
        imgs_with_boxes.append(img_res)
    imgs = torch.stack(imgs_with_boxes).type(torch.float32)/256
    return make_grid(imgs, nrow=8, normalize=True)

class Boxes:
    """ Manupulating Boxes
    """
    def __init__(self, boxes):
        """
        :param boxes: list of Box object or list of tuple (x, y, w, h)
        """
        if isinstance(boxes[0], Box):
            self.boxes = boxes
        else:
            self.boxes = [Box(box) for box in boxes]
        self.idxes = np.arange(len(boxes))
    
    def __len__(self):
        return self.idxes.shape[0]
    
    def rm_parent_boxes(self):
        """
        When a box contains any other boxes, treat it as a parent box.
        REMOVE all the parent boxes.
        """
        to_delete_idxes = []
        for i in range(len(self)):
            for j in range(len(self)):
                if i == j: continue
                if self.boxes[i].contains(self.boxes[j]):
                   to_delete_idxes.append(i)
                   break
        self.idxes = np.delete(self.idxes, to_delete_idxes)

    def find_non_overlap_boxes(self):
        """find boxes which are not overlap with others
        """
        non_overlap_boxes = []
        for i in range(len(self)):
            box_idx = self.idxes[i]
            if i == 0:
                non_overlap_boxes.append(box_idx)
            else:
                for box_jdx in non_overlap_boxes:
                    if Box.is_overlap(self.boxes[box_idx], self.boxes[box_jdx]):
                        break
                non_overlap_boxes.append(box_idx)
        return non_overlap_boxes
    
    def rm_overlapping_boxes(self):
        self.rm_parent_boxes()
        non_overlap_boxes = self.find_non_overlap_boxes()
        self.idxes = np.array(non_overlap_boxes)

    def visualize_all_boxes(self, path):
        """Create a white 100x100 canvas, 
        draw all the self.boxes as hollow white rectangles one by one with (box.x, box.y, box.w, box.h) as coordinates.
        Save the canvas to path.
        """
        print(f"Visualizing {len(self)} boxes...")
        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        for i in self.idxes:
            box = self.boxes[i]
            # turn ratio into int pixel
            x, y, w, h = box.x*100, box.y*100, box.w*100, box.h*100
            cv2.rectangle(canvas, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 0), thickness=1)
        cv2.imwrite(path, canvas)

    def expand(self, scale):
        """
        :param scale: float, scale factor
        """
        for i in range(len(self)):
            self.boxes[i].expand(scale)

    def rm_small_boxes(self, min_size=0.02):
        """
        :param min_size: float, minimum size of box
        """
        to_delete_idxes = []
        for i in range(len(self)):
            if self.boxes[i].w < min_size or self.boxes[i].h < min_size:
                to_delete_idxes.append(i)
        self.idxes = np.delete(self.idxes, to_delete_idxes)

class Box:
    """ Manuplating Box
    """
    def __init__(self, box):
        """
        :param box: list of box (x, y, w, h)
        """
        self.box = np.array(box)
        self.x, self.y, self.w, self.h = self.box[0], self.box[1], self.box[2], self.box[3]
        self.x1, self.y1, self.x2, self.y2 = self.x, self.y, self.x+self.w, self.y+self.h
    
    @staticmethod
    def is_overlap(box1, box2):
        """
        :param box1: Box object
        :param box2: Box object
        :return: True if overlap, False otherwise
        """
        return not (box1.x2 < box2.x1 or box1.x1 > box2.x2 or box1.y2 < box2.y1 or box1.y1 > box2.y2)
    
    @staticmethod
    def is_contain(box1, box2):
        """
        :param box1: Box object
        :param box2: Box object
        :return: True if box1 contains box2, False otherwise
        """
        return box1.x1 <= box2.x1 and box1.x2 >= box2.x2 and box1.y1 <= box2.y1 and box1.y2 >= box2.y2

    def contains(self, box):
        """
        :param box: Box object
        :return: True self.box contains box, False otherwise
        """
        return self.is_contain(self, box)

    def expand(self, scale):
        """
        :param scale: float, scale factor with center the same as self.box
        """
        center = np.array([self.x+self.w/2, self.y+self.h/2])
        self.x, self.y, self.w, self.h = center[0]-self.w/2*scale, center[1]-self.h/2*scale, self.w*scale, self.h*scale
        self.x1, self.y1, self.x2, self.y2 = self.x, self.y, self.x+self.w, self.y+self.h

def get_new_dataset(name):
    import fiftyone as fo
    try:
        dataset = fo.Dataset(name)
        print(f"create new dataset: {name}")
    except:
        dataset = fo.load_dataset(name)
        dataset.delete()
        print(f"delete and create new dataset: {name}")
        dataset = fo.Dataset(name)
    return dataset

def create_tiny_dataset(source_ds, sample_num=1000, name="tiny_dataset", force_create=False):
    """Create a tiny version of input fiftyone.ataset source_ds, with sample_num samples.
    Args:
        source_ds (fiftyone.Dataset): source dataset
        sample_num (int): number of samples to be created
        name (str): name of the new dataset
    Returns:
        fiftyone.Dataset: new dataset
    """
    import fiftyone as fo
    if name in fo.list_datasets(): 
        if force_create:
            fo.delete_dataset(name)
        else:
            return fo.load_dataset(name)
    res_ds = get_new_dataset(name)
    res_ds.persistent = True
    for s in tqdm(source_ds):
        res_ds.add_sample(s)
        if len(res_ds) >= sample_num: break
    return res_ds
    
def ratio_lengths_support(lengths_in, total_length):
    if type(lengths_in[0]) == float:
        lengths = [int(r*total_length) for r in lengths_in]
        lengths[0] = total_length - sum(lengths[1:]) # adjust to make length sum equal to len(dataset)
    else:
        lengths = lengths_in
    return lengths

def IoU(a, b):
    """calculate IoU"""
    x1, y1, x2, y2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    x3, y3, x4, y4 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))
    overlap_area = x_overlap * y_overlap
    a_area = (x2-x1)*(y2-y1)
    b_area = (x4-x3)*(y4-y3)
    return overlap_area / (a_area + b_area - overlap_area)

"""MT"""
