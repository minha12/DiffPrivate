from diffusers import StableDiffusionPipeline, DDIMScheduler
from src.attCtr import AttentionControlEdit
import src.diffprivate as diffprivate
from PIL import Image
import os
import glob
from natsort import ns, natsorted
import hydra
from omegaconf import DictConfig

from src.utils import seed_torch

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set random seed first
    seed_torch(42)
    
    assert (
        cfg.diffusion.res % 32 == 0 and cfg.diffusion.res >= 96
    ), "Please ensure the input resolution be a multiple of 32 and also >= 96."

    save_dir = cfg.paths.save_dir  # Where to save the adversarial examples, and other results.
    os.makedirs(save_dir, exist_ok=True)
    
    images_root = cfg.paths.images_root  # The clean images' root directory.

    pretrained_diffusion_path = cfg.paths.pretrained_diffusion_path

    ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to(
        "cuda:0"
    )
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)

    "Attack a subset images"
    all_images = glob.glob(os.path.join(images_root, "*"))
    # Filter only .png and .jpg files
    all_images = [img for img in all_images if img.lower().endswith((".png", ".jpg"))]
    all_images = natsorted(all_images, alg=ns.PATH)

    blackbox_frs_distances = []
    blackbox_frs_recognition = []

    for ind, image_path in enumerate(all_images):
        image = Image.open(image_path).convert("RGB")
        # get image name without extension
        image_name = os.path.basename(image_path).split(".")[0]
        image.save(os.path.join(cfg.paths.save_dir, image_name + "_originImage.png"))

        controller = AttentionControlEdit(
            cfg.diffusion.diffusion_steps,
            cfg.diffusion.self_replace_steps,  # Wrap in list to match expected format
            cfg.diffusion.res,
        )

        _, clean_acc, adv_acc = diffprivate.protect(
            model=ldm_stable,
            controller=controller,
            args=cfg,
            image_path=image_path,
            save_path=os.path.join(save_dir, image_name),
        )

        blackbox_frs_distances.append(clean_acc)
        blackbox_frs_recognition.append(adv_acc)

if __name__ == "__main__":
    main()
