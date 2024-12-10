import sys
import os
sys.path.append("./src/DiffAE")

import hydra
from omegaconf import DictConfig

from src.DiffAE.dataset import CelebAttrDataset
from src.diffprivate_edit import _load_cls_model, _load_diffae, attack, attack_sem
from src.DiffAE.general_config import ffhq256_autoenc
from src.utils import preprocess

import torch
import torch.nn.functional as F
from PIL import Image
from natsort import natsorted, ns

import glob
import math
import os

def run_protect(cfg):

    device = "cuda:0"
    model = _load_diffae()
    cls_model = _load_cls_model()
    conf = ffhq256_autoenc()
    cfg.img_size = conf.img_size
    image_paths = glob.glob(os.path.join(cfg.src_path, "*"))
    image_paths = [img for img in image_paths if img.lower().endswith((".png", ".jpg"))]
    image_paths = natsorted(image_paths, alg=ns.PATH)
    tar_img = Image.open(cfg.tar_path).convert("RGB")
    tar_img = preprocess(tar_img, conf.img_size)
    for ind, image_path in enumerate(image_paths):
        # Encoding (remains mostly unchanged)
        src_img = Image.open(image_path).convert("RGB")
        src_img = preprocess(src_img, conf.img_size)
        with torch.no_grad():
            print("Encoding ...")
            cond = model.encode(src_img.to(device))
            xT = model.encode_stochastic(src_img.to(device), cond, T=250)
        if cfg.semantic_editting:
            # Attribute modification
            cls_id = CelebAttrDataset.cls_to_id[cfg.tar_attr]
            with torch.no_grad():
                cond_new = cls_model.normalize(cond)
                cond_new += (
                    cfg.strength
                    * math.sqrt(512)
                    * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
                )
                cond_new = cls_model.denormalize(cond_new)

            # Optimization using attack function
            file_name = os.path.basename(image_path).split(".")[0]
            save_path = os.path.join(cfg.results_dir, file_name)
            print(save_path)
            img_gen = attack_sem(
                cfg, model, cond, xT, cond_new, src_img, tar_img, save_path
            )
        else:
            #  Optimization using attack function
            file_name = os.path.basename(image_path).split(".")[0]
            save_path = os.path.join(cfg.results_dir, file_name)
            img_gen = attack(cfg, model, cond, xT, src_img, tar_img, save_path)

    return img_gen

@hydra.main(version_base=None, config_path="configs", config_name="config_dpe")
def main(cfg: DictConfig):
    print(cfg)
    run_protect(cfg)

if __name__ == "__main__":
    main()