from datetime import datetime
import json
import torch
import src.other_attacks.pgd as pgd
from PIL import Image
import numpy as np
import os
import glob

import random

from natsort import ns, natsorted
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--save_dir",
    default="output",
    type=str,
    help="Where to save the adversarial examples, and other results",
)
parser.add_argument(
    "--pgd_alpha",
    type=float,
    default=5e-3,
    help="The alpha for PGD attack",
)
parser.add_argument(
    "--pgd_eps",
    type=float,
    default=5e-2,
    help="The noise budget for pgd.",
)
parser.add_argument(
    "--targeted_attack",
    default=True,
    type=bool,
    help="Whether to perform targeted attack",
)

parser.add_argument(
    "--ensemble",
    default=False,
    type=bool,
    help="Whether to use ensemble of ID models",
)
parser.add_argument(
    "--list_attacker_models",
    default=["irse50", "ir152", "cur_face"],
    nargs="+",
    help="The surrogate models: irse50, facenet, mobile_face, ir152, cur_face",
)
parser.add_argument(
    "--ensemble_mode",
    default="mean",
    type=str,
    help="The mode for ensemble: min, max, mean",
)

parser.add_argument(
    "--attacker_model",
    default="irse50",
    type=str,
    help="The surrogate model from which the adversarial examples are crafted",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.421,
    help="Threshold for facial recognition",
)
parser.add_argument(
    "--victim_model",
    default="facenet",
    type=str,
    help="The victim model for facial recognition",
)
parser.add_argument(
    "--victim_threshold",
    type=float,
    default=0.36,
    help="Threshold for facial recognition",
)
parser.add_argument(
    "--target_image",
    type=str,
    default="./data/target_imgs/00187.jpg",
    help="The target image for the attack",
)

parser.add_argument(
    "--overhead",
    type=float,
    default=0.1,
    help="Overhead for the attack",
)

parser.add_argument(
    "--max_iter",
    type=int,
    default=1000,
    help="Maximum number of iterations",
)
parser.add_argument(
    "--images_root",
    default="./data/demo/images",
    type=str,
    help="The clean images root directory",
)

parser.add_argument(
    "--iterations", default=150, type=int, help="Iterations of optimizing the adv_image"
)
parser.add_argument(
    "--res", default=256, type=int, help="Input image resized resolution"
)
parser.add_argument(
    "--attack_loss_weight", default=1, type=float, help="attack loss weight factor"
)


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


seed_torch(42)


def run_protect(
    image_path,
    save_dir=r"",
    res=224,
    iterations=30,
    args=None,
):

    adv_image, distances, recognitions = pgd.attack(
        image_path=image_path,
        save_path=save_dir,
        res=res,
        iterations=iterations,
        args=args,
    )

    return adv_image, distances, recognitions


def get_datetime_prefix():
    # Get current date and time
    now = datetime.now()

    prefix = now.strftime("%Y%m%d-%H%M%S")

    return prefix


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


if __name__ == "__main__":
    args = parser.parse_args()
    assert (
        args.res % 32 == 0 and args.res >= 96
    ), "Please ensure the input resolution be a multiple of 32 and also >= 96."

    iterations = args.iterations  # Iterations of optimizing the adv_image.
    res = args.res  # Input image resized resolution.

    save_dir = (
        args.save_dir
    )  # Where to save the adversarial examples, and other results.
    os.makedirs(save_dir, exist_ok=True)

    "If you set 'is_test' to True, please turn 'images_root' to the path of the output results' path."
    images_root = args.images_root  # The clean images' root directory.

    "Attack a subset images"
    all_images = glob.glob(os.path.join(images_root, "*"))
    all_images = natsorted(all_images, alg=ns.PATH)

    adv_images = []
    images = []
    blackbox_frs_distances = []
    blackbox_frs_recognition = []
    adv_all_acc = 0

    for ind, image_path in enumerate(all_images):
        image = Image.open(image_path).convert("RGB")
        image.save(
            os.path.join(args.save_dir, str(ind).rjust(4, "0") + "_originImage.png")
        )
        adv_image, distance_from_blackbox_dict, recognition_from_blackbox_dict = (
            run_protect(
                image_path,
                res=res,
                iterations=iterations,
                save_dir=os.path.join(save_dir, str(ind).rjust(4, "0")),
                args=args,
            )
        )
        blackbox_frs_distances.append(distance_from_blackbox_dict)
        blackbox_frs_recognition.append(recognition_from_blackbox_dict)

    attackers = "_".join(args.list_attacker_models)

    prefix = get_datetime_prefix()
    dataset_name = args.images_root.split("/")[-1]
    # print(blackbox_frs_distances)
    # print(blackbox_frs_recognition)
    # save a text file to store args and other information
    args_dict = vars(args)
    with open(
        os.path.join(
            save_dir,
            f"{prefix}_args-{dataset_name}_targeted-{str(args.targeted_attack)}_attacker-{attackers}_victim-{args.victim_model}.json",
        ),
        "w",
    ) as json_file:
        json.dump(args_dict, json_file, indent=4)

    # Combine dictionaries
    combine_then_save_json(
        blackbox_frs_distances,
        file_name=os.path.join(
            save_dir,
            f"{prefix}_distance-{dataset_name}_targeted-{str(args.targeted_attack)}_attacker-{attackers}_victim-{args.victim_model}.json",
        ),
    )
    combine_then_save_json(
        blackbox_frs_recognition,
        file_name=os.path.join(
            save_dir,
            f"{prefix}_recognition-{dataset_name}_targeted-{str(args.targeted_attack)}_attacker-{attackers}_victim-{args.victim_model}.json",
        ),
    )
