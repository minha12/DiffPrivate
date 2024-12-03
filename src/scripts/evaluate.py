import torch
import numpy as np
from PIL import Image
from src.criteria.id_loss import IdLost
import argparse
import os
from tqdm import tqdm

threshold_dict = {
    "irse50": 0.4,
    "ir152": 0.4277,
    "facenet": 0.33519999999999994,
    "cur_face": 0.4332,
    "mobile_face": 0.3875,
}


def arg_parser():

    parser = argparse.ArgumentParser(description="DiffPure")
    parser.add_argument(
        "--data_folder", type=str, default="./output", help="Data folder"
    )
    parser.add_argument(
        "--ignore_extension",
        type=bool,
        default=False,
        help="Whether to ignore file extension",
    )
    parser.add_argument(
        "--adv_folder", type=str, default="./output", help="Adversarial data folder"
    )
    parser.add_argument(
        "--clean_folder", type=str, default="./output", help="Clean data folder"
    )
    parser.add_argument(
        "--folder_type",
        type=str,
        default="single",
        help="Type of folder (single or separate, fawkes)",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="0.412",
        help="Threshold for facial recognition",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
    )

    return parser


def find_image_pairs(folder_path):
    # Suffixes to identify the original and adversarial images
    suffix_origin = "_originImage.png"
    suffix_adv = "_adv_image.png"

    # Dictionary to hold the pairs
    image_pairs = {}

    # List all files in the given folder
    for file in os.listdir(folder_path):
        # print(file)
        if file.endswith(suffix_origin) or file.endswith(suffix_adv):
            # Extract prefix (assuming fixed length prefixes, e.g., "0000")
            prefix = file.split("_")[0]

            if prefix not in image_pairs:
                image_pairs[prefix] = {"origin": None, "adv": None}

            # Assign the file to the correct part of the pair
            if file.endswith(suffix_origin):
                image_pairs[prefix]["origin"] = os.path.join(folder_path, file)
            elif file.endswith(suffix_adv):
                image_pairs[prefix]["adv"] = os.path.join(folder_path, file)
    print(image_pairs)
    # Filter out incomplete pairs
    complete_pairs = [
        (pair["origin"], pair["adv"])
        for pair in image_pairs.values()
        if pair["origin"] and pair["adv"]
    ]
    # print(complete_pairs)
    return complete_pairs


def find_matching_pairs_fawkes(original_folder, cloaked_folder):
    # Dictionary to store matching pairs
    matching_pairs = {}

    # Gather all filenames in the original folder
    original_files = {
        f.split(".")[0]: os.path.join(original_folder, f)
        for f in os.listdir(original_folder)
        if f.endswith(".jpg")
    }

    # Loop through all files in the cloaked folder
    for cloaked_file in os.listdir(cloaked_folder):
        if cloaked_file.endswith("_high_cloaked.png"):
            # Extract the prefix (number before '_high_cloaked.png')
            prefix = cloaked_file.split("_")[0]

            # Check if this prefix matches any file in the original folder
            if prefix in original_files:
                cloaked_path = os.path.join(cloaked_folder, cloaked_file)
                matching_pairs[prefix] = {
                    "origin": original_files[prefix],
                    "cloaked": cloaked_path,
                }

    # Convert dictionary values to a list of tuples for the final output
    complete_pairs = [
        (value["origin"], value["cloaked"]) for key, value in matching_pairs.items()
    ]

    return complete_pairs


# Helper function to either return the filename as-is or without its extension
def process_filename(filename, ignore_extensions):
    return os.path.splitext(filename)[0] if ignore_extensions else filename


# Dictionary to store image pairs with the processed filename as the key
image_pairs = {}


# Function to process and gather filenames based on the ignore_extensions flag
def gather_files(folder, ignore_extensions):
    files = os.listdir(folder)
    if ignore_extensions:
        # Remove extensions for comparison
        processed_files = {
            process_filename(f, True): f for f in files if f.endswith((".jpg", ".png"))
        }
    else:
        # Keep original filenames for comparison
        processed_files = {f: f for f in files if f.endswith((".jpg", ".png"))}
    return processed_files


def find_image_pairs_from_folder(folder_original, folder_adv, ignore_extensions=True):

    # Gather all image filenames from both folders with or without extensions
    original_files = gather_files(folder_original, ignore_extensions)

    adv_files = gather_files(folder_adv, ignore_extensions)

    # Find matching filenames (with or without extensions) in both folders
    matching_filenames = set(original_files.keys()).intersection(adv_files.keys())

    # Populate the dictionary with pairs
    for base_name in matching_filenames:
        original_path = os.path.join(folder_original, original_files[base_name])
        adv_path = os.path.join(folder_adv, adv_files[base_name])
        image_pairs[base_name] = {"origin": original_path, "adv": adv_path}

    # Convert dictionary values to a list of tuples for the final output
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
    img = Image.open(path)
    return preprocess(img)


def run(args):
    if args.folder_type == "separate":
        pairs = find_image_pairs_from_folder(
            args.data_folder, args.adv_folder, args.ignore_extension
        )
    elif args.folder_type == "fawkes":
        pairs = find_matching_pairs_fawkes(args.data_folder, args.adv_folder)
    elif args.folder_type == "single":
        print("Using single folder")
        pairs = find_image_pairs(args.data_folder)

    model_names = [key for key in threshold_dict]

    frs_models = [IdLost(model_name) for model_name in model_names]
    # print(pairs)
    successes = []
    distances_collection = []
    for org_path, adv_path in tqdm(pairs):
        adv_img = load_and_preprocess_image(adv_path)
        org_img = load_and_preprocess_image(org_path)

        distances = np.array(
            [
                frs_models[i](org_img, adv_img).detach().cpu().numpy()
                for i in range(len(threshold_dict))
            ]
        )
        distances_collection.append(distances)
        sucess = distances > np.array([threshold_dict[model] for model in model_names])
        successes.append(sucess)

    success_rate = np.array(successes).astype(int).mean(axis=0)
    avg_dist = np.array(distances_collection).mean(axis=0)
    print(f"Success rate: {success_rate}")
    print(f"Average distance: {avg_dist}")
    # set log file name to the name of data_folder without the base directory
    # create log dir if it does not exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if args.folder_type == "separate" or args.folder_type == "fawkes":
        log_path = os.path.join(
            args.log_dir, f"{os.path.basename(os.path.normpath(args.adv_folder))}.txt"
        )
    if args.folder_type == "single":
        log_path = os.path.join(
            args.log_dir, f"{os.path.basename(os.path.normpath(args.data_folder))}.txt"
        )

    # add the model names and success rate to the log file
    with open(log_path, "a") as f:
        f.write(f"Model names: {model_names}\n")
        f.write(f"Success rate: {success_rate}\n")
        f.write(f"Average distance: {avg_dist}\n")

    return success_rate


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    run(args)
