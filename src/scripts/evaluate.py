import torch
import numpy as np
from PIL import Image
from criteria.id_loss import IdLost
from criteria.lpips.lpips import LPIPS
import hydra
from omegaconf import DictConfig
import os
from tqdm import tqdm

def find_image_pairs(folder_path):
    suffix_origin = "_originImage.png"
    suffix_adv = "_adv_image.png"
    image_pairs = {}

    for file in os.listdir(folder_path):
        if file.endswith(suffix_origin) or file.endswith(suffix_adv):
            prefix = file.split("_")[0]

            if prefix not in image_pairs:
                image_pairs[prefix] = {"origin": None, "adv": None}

            if file.endswith(suffix_origin):
                image_pairs[prefix]["origin"] = os.path.join(folder_path, file)
            elif file.endswith(suffix_adv):
                image_pairs[prefix]["adv"] = os.path.join(folder_path, file)

    complete_pairs = [
        (pair["origin"], pair["adv"])
        for pair in image_pairs.values()
        if pair["origin"] and pair["adv"]
    ]
    return complete_pairs

def find_matching_pairs_fawkes(original_folder, cloaked_folder):
    matching_pairs = {}
    original_files = {
        f.split(".")[0]: os.path.join(original_folder, f)
        for f in os.listdir(original_folder)
        if f.endswith(".jpg")
    }

    for cloaked_file in os.listdir(cloaked_folder):
        if cloaked_file.endswith("_high_cloaked.png"):
            prefix = cloaked_file.split("_")[0]

            if prefix in original_files:
                cloaked_path = os.path.join(cloaked_folder, cloaked_file)
                matching_pairs[prefix] = {
                    "origin": original_files[prefix],
                    "adv": cloaked_path,
                }

    complete_pairs = [
        (value["origin"], value["adv"]) for key, value in matching_pairs.items()
    ]
    return complete_pairs

def process_filename(filename, ignore_extensions):
    return os.path.splitext(filename)[0] if ignore_extensions else filename

def gather_files(folder, ignore_extensions):
    files = os.listdir(folder)
    if ignore_extensions:
        processed_files = {
            process_filename(f, True): f for f in files if f.lower().endswith((".jpg", ".png"))
        }
    else:
        processed_files = {f: f for f in files if f.lower().endswith((".jpg", ".png"))}
    return processed_files

def find_image_pairs_from_folder(folder_original, folder_adv, ignore_extensions):
    image_pairs = {}
    original_files = gather_files(folder_original, ignore_extensions)
    adv_files = gather_files(folder_adv, ignore_extensions)

    matching_filenames = set(original_files.keys()).intersection(adv_files.keys())

    for base_name in matching_filenames:
        original_path = os.path.join(folder_original, original_files[base_name])
        adv_path = os.path.join(folder_adv, adv_files[base_name])
        image_pairs[base_name] = {"origin": original_path, "adv": adv_path}

    complete_pairs = [
        (value["origin"], value["adv"]) for key, value in image_pairs.items()
    ]
    return complete_pairs

def preprocess(image, res=256):
    """Preprocess the image."""
    image = image.resize((res, res))
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0

def load_and_preprocess_image(path):
    """Load and preprocess an image."""
    img = Image.open(path).convert('RGB')
    return preprocess(img)

@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig):
    # Load thresholds from config
    threshold_dict = {key: float(value) for key, value in cfg.thresholds.items()}

    if cfg.evaluation.folder_type == "separate":
        pairs = find_image_pairs_from_folder(
            cfg.evaluation.clean_folder,
            cfg.evaluation.adv_folder,
            cfg.evaluation.ignore_extension,
        )
    elif cfg.evaluation.folder_type == "fawkes":
        pairs = find_matching_pairs_fawkes(
            cfg.evaluation.clean_folder,
            cfg.evaluation.adv_folder,
        )
    elif cfg.evaluation.folder_type == "single":
        print("Using single folder")
        pairs = find_image_pairs(cfg.evaluation.data_folder)
    else:
        raise ValueError("Invalid folder_type provided.")

    model_names = list(threshold_dict.keys())
    frs_models = [IdLost(model_name) for model_name in model_names]
    dist_model = LPIPS().cuda()

    successes = []
    distances_collection = []
    lpips_distances = []

    for org_path, adv_path in tqdm(pairs):
        adv_img = load_and_preprocess_image(adv_path)
        org_img = load_and_preprocess_image(org_path)

        distances = np.array([
            frs_models[i](org_img, adv_img).detach().cpu().numpy()
            for i in range(len(model_names))
        ]).flatten()
        distances_collection.append(distances)

        success = distances > np.array([threshold_dict[model] for model in model_names])
        successes.append(success)

        lpips_distance = dist_model(org_img, adv_img).item()
        lpips_distances.append(lpips_distance)

    successes = np.array(successes).astype(int)
    success_rates = successes.mean(axis=0)
    avg_distances = np.array(distances_collection).mean(axis=0)
    avg_lpips = np.mean(lpips_distances)

    headers = ["Model", "Success Rate", "Average Distance"]
    data = [
        [model_names[i], f"{success_rates[i]:.4f}", f"{avg_distances[i]:.4f}"]
        for i in range(len(model_names))
    ]

    print("{:<15} {:<15} {:<17}".format(*headers))
    print("-" * 47)
    for row in data:
        print("{:<15} {:<15} {:<17}".format(*row))

    print(f"\nAverage LPIPS distance: {avg_lpips:.4f}")

    if not os.path.exists(cfg.evaluation.log_dir):
        os.makedirs(cfg.evaluation.log_dir)

    if cfg.evaluation.folder_type in ["separate", "fawkes"]:
        log_filename = f"{os.path.basename(os.path.normpath(cfg.evaluation.adv_folder))}.txt"
    elif cfg.evaluation.folder_type == "single":
        log_filename = f"{os.path.basename(os.path.normpath(cfg.evaluation.data_folder))}.txt"
    else:
        log_filename = "evaluation_log.txt"

    log_path = os.path.join(cfg.evaluation.log_dir, log_filename)

    with open(log_path, "a") as f:
        f.write("{:<15} {:<15} {:<17}\n".format(*headers))
        f.write("-" * 47 + "\n")
        for row in data:
            f.write("{:<15} {:<15} {:<17}\n".format(*row))
        f.write(f"\nAverage LPIPS distance: {avg_lpips:.4f}\n")

if __name__ == "__main__":
    main()
