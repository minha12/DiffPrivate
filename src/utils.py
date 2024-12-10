from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from multiprocessing import Process, Queue
import json
import os
import random
import shutil

import cv2
import eagerpy as ep
import lmdb
import numpy as np
import torch
import torch.backends.cudnn
from PIL import Image
from typing import Tuple, TypeVar
from torch.utils.data import Dataset

def aggregate_attention(
    prompts,
    attention_store,
    res: int,
    from_where,
    is_cross: bool,
    select: int,
    is_cpu=True,
):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[
                    select
                ]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu() if is_cpu else out


def show_cross_attention(
    prompts,
    tokenizer,
    attention_store,
    res: int,
    from_where,
    select: int = 0,
    save_path=None,
):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(
        prompts, attention_store, res, from_where, True, select
    )
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.detach().cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    for image in images:
        print(image.shape)
    render_image_grid(np.stack(images, axis=0), save_path=save_path)


def render_image_grid(images, num_rows=1, offset_ratio=0.02, save_path=None, show=False):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if show:
        pil_img.show()
    if save_path is not None:
        pil_img.save(save_path)


def show_self_attention_comp(
    prompts,
    attention_store,
    res: int,
    from_where,
    max_com=7,
    select: int = 0,
    save_path=None,
):
    attention_maps = (
        aggregate_attention(prompts, attention_store, res, from_where, False, select)
        .numpy()
        .reshape((res**2, res**2))
    )
    u, s, vh = np.linalg.svd(
        attention_maps - np.mean(attention_maps, axis=1, keepdims=True)
    )
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    save_images(np.concatenate(images, axis=1), save_path=save_path)


def text_under_image(
    image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
):
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def save_images(image, save_path=None):
    # Remove unnecessary dimensions (assuming the first image in batch if batch dimension exists)
    if len(image.shape) > 3:
        image = image.squeeze()

    # Convert to uint8 if not already
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    pil_img = Image.fromarray(image)

    if save_path is not None:
        pil_img.save(save_path)


def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def save_results(image, save_path, init_image):
    real = (
        (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).squeeze().cpu().numpy()
    )

    perturbed = image.permute(0, 2, 3, 1).squeeze().cpu().numpy()

    side_by_side_image = np.concatenate([real, perturbed], axis=1)
    save_images(
        side_by_side_image,
        save_path + "_diff_image_{}.png".format("ATKSuccess"),
    )

    save_images(perturbed, save_path + "_adv_image.png")

    diff = perturbed - real
    diff = (diff - diff.min()) / (diff.max() - diff.min())

    save_images(diff.clip(0, 1), save_path + "_diff_relative.png")

    diff = np.abs(perturbed - real)
    save_images(diff.clip(0, 1), save_path + "_diff_absolute.png")

    return image


def post_process(img):
    out_img = (img / 2 + 0.5).clamp(0, 1)
    # out_img = out_img.permute(0, 2, 3, 1)
    # out_img = out_img.permute(0, 3, 1, 2)
    return out_img


def tensor2pil2tensor(res, out_image):
    np_image = (
        out_image.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy() * 255
    ).astype(np.uint8)
    pil_img = Image.fromarray(np_image)
    reload_image = preprocess(pil_img, res)
    return reload_image


def combine_then_save_json(list_of_dicts, file_name="distances.json"):
    combined_dicts = {}
    # Combine dictionaries and convert numpy arrays to lists
    for d in list_of_dicts:
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if key not in combined_dicts:
                combined_dicts[key] = [value]
            else:
                combined_dicts[key].append(value)

    # Temporary dictionary for storing averages
    averages = {}

    # Calculate the mean for each key and store it in the temporary dictionary
    for key, values in combined_dicts.items():
        if all(isinstance(v, (int, float, list)) for v in values):
            values = [
                np.array(v) for v in values if isinstance(v, list) or np.isscalar(v)
            ]
            mean_value = np.mean(values, axis=0)
            if isinstance(mean_value, np.ndarray):
                mean_value = mean_value.tolist()
            averages[f"avg_{key}"] = mean_value
        else:
            print(f"Skipping mean calculation for {key} due to non-numeric values.")

    # Update the original dictionary with the averages
    combined_dicts.update(averages)

    # Save the updated dictionary to a JSON file
    with open(file_name, "w") as json_file:
        json.dump(combined_dicts, json_file, indent=4)


def get_datetime_prefix():
    # Get current date and time
    now = datetime.now()

    prefix = now.strftime("%Y%m%d-%H%M%S")

    return prefix


def seed_torch(seed=42):
    """For reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def flatten(x: ep.Tensor, keep: int = 1) -> ep.Tensor:
    return x.flatten(start=keep)

def atleast_kd(x: ep.Tensor, k: int) -> ep.Tensor:
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)

T = TypeVar("T")

class Distance(ABC):
    @abstractmethod
    def __call__(self, reference: T, perturbed: T) -> T: ...

    @abstractmethod
    def clip_perturbation(self, references: T, perturbed: T, epsilon: float) -> T: ...

class LpDistance(Distance):
    def __init__(self, p: float):
        self.p = p

    def __repr__(self) -> str:
        return f"LpDistance({self.p})"

    def __str__(self) -> str:
        return f"L{self.p} distance"

    def __call__(self, references: T, perturbed: T) -> T:
        (x, y), restore_type = ep.astensors_(references, perturbed)
        norms = ep.norms.lp(flatten(y - x), self.p, axis=-1)
        return restore_type(norms)

    def clip_perturbation(self, references: T, perturbed: T, epsilon: float) -> T:
        (x, y), restore_type = ep.astensors_(references, perturbed)
        p = y - x
        if self.p == ep.inf:
            clipped_perturbation = ep.clip(p, -epsilon, epsilon)
            return restore_type(x + clipped_perturbation)
        norms = ep.norms.lp(flatten(p), self.p, axis=-1)
        norms = ep.maximum(norms, 1e-12)  # avoid division by zero
        factor = epsilon / norms
        factor = ep.minimum(1, factor)
        if self.p == 0:
            if (factor == 1).all():
                return perturbed
            raise NotImplementedError("reducing L0 norms not yet supported")
        factor = atleast_kd(factor, x.ndim)
        clipped_perturbation = factor * p
        return restore_type(x + clipped_perturbation)

def save_edited_results(image, model_name, save_path, init_image):
    real = (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    perturbed = image
    image = (perturbed * 255).astype(np.uint8)
    render_image_grid(
        np.concatenate([real, perturbed]) * 255,
        show=False,
        save_path=save_path
        + "_diff_{}_image_{}.png".format(
            model_name, "ATKSuccess"
        ),
    )
    render_image_grid(perturbed * 255, show=False, save_path=save_path + "_adv_image.png")

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))

    print(
        "L1: {}\tL2: {}\tLinf: {}".format(
            L1(real, perturbed), L2(real, perturbed), Linf(real, perturbed)
        )
    )

    diff = perturbed - real
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255

    render_image_grid(
        diff.clip(0, 255), show=False, save_path=save_path + "_diff_relative.png"
    )

    diff = (np.abs(perturbed - real) * 255).astype(np.uint8)
    render_image_grid(
        diff.clip(0, 255), show=False, save_path=save_path + "_diff_absolute.png"
    )

    return image

def preprocess_image_imagenet_norm(img):
    out_img = (img / 2 + 0.5).clamp(0, 1)
    out_img = out_img.permute(0, 2, 3, 1)
    mean = torch.as_tensor(
        [0.485, 0.456, 0.406], dtype=out_img.dtype, device=out_img.device
    )
    std = torch.as_tensor(
        [0.229, 0.224, 0.225], dtype=out_img.dtype, device=out_img.device
    )
    out_img = out_img[:, :, :].sub(mean).div(std)
    out_img = out_img.permute(0, 3, 1, 2)
    return out_img

def tensor2numpy(image):
    image = image.detach().permute(0, 2, 3, 1)
    image = image.cpu().numpy()
    return image

def convert(x, format, quality=100):
    torch.set_num_threads(1)
    buffer = BytesIO()
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    x = x.to(torch.uint8)
    x = x.numpy()
    img = Image.fromarray(x)
    img.save(buffer, format=format, quality=quality)
    val = buffer.getvalue()
    return val

@contextmanager
def nullcontext():
    yield

class _WriterWroker(Process):
    def __init__(self, path, format, quality, zfill, q):
        super().__init__()
        if os.path.exists(path):
            shutil.rmtree(path)
        self.path = path
        self.format = format
        self.quality = quality
        self.zfill = zfill
        self.q = q
        self.i = 0

    def run(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with lmdb.open(self.path, map_size=1024**4, readahead=False) as env:
            while True:
                job = self.q.get()
                if job is None:
                    break
                with env.begin(write=True) as txn:
                    for x in job:
                        key = f"{str(self.i).zfill(self.zfill)}".encode("utf-8")
                        x = convert(x, self.format, self.quality)
                        txn.put(key, x)
                        self.i += 1
            with env.begin(write=True) as txn:
                txn.put("length".encode("utf-8"), str(self.i).encode("utf-8"))

class LMDBImageWriter:
    def __init__(self, path, format="webp", quality=100, zfill=7) -> None:
        self.path = path
        self.format = format
        self.quality = quality
        self.zfill = zfill
        self.queue = None
        self.worker = None

    def __enter__(self):
        self.queue = Queue(maxsize=3)
        self.worker = _WriterWroker(
            self.path, self.format, self.quality, self.zfill, self.queue
        )
        self.worker.start()

    def put_images(self, tensor):
        self.queue.put(tensor.cpu())

    def __exit__(self, *args, **kwargs):
        self.queue.put(None)
        self.queue.close()
        self.worker.join()

class LMDBImageReader(Dataset):
    def __init__(self, path, zfill: int = 7):
        self.zfill = zfill
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f"{str(index).zfill(self.zfill)}".encode("utf-8")
            img_bytes = txn.get(key)
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return img
