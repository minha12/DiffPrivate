import json
import logging
import os
import torch
from PIL import Image
from tqdm import tqdm

from src.utils import tensor2pil2tensor

from src.utils import preprocess, save_results

from src.criteria.id_loss import IdLost, EnsembleIdLost


def normalizedIdLoss(x, threshold=0.412, apply_sigmoid=True):
    # return sigmoid(x / threshold)
    if apply_sigmoid:
        return torch.sigmoid(x / threshold)
    else:
        return x / threshold


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


threshold_dict = {
    "irse50": 0.412,
    "ir152": 0.42,
    "facenet": 0.36,
    "cur_face": 0.43,
    "mobile_face": 0.425,
}


@torch.enable_grad()
def attack(
    image_path=None,
    save_path=r"",
    res=224,
    iterations=30,
    verbose=True,
    args=None,
):
    image = Image.open(image_path).convert("RGB")
    if args.ensemble:
        idLossFunc = EnsembleIdLost(args.list_attacker_models, args.ensemble_mode)
    else:
        idLossFunc = IdLost(args.attacker_model)
    frs_model = IdLost(args.victim_model)

    init_image = preprocess(image, res)
    # checking value range of init_image
    print(init_image.min(), init_image.max())
    init_image.requires_grad = False
    # pertubed image is random tensor in range -1, 1 - same size as original image
    # pertubed_image = torch.rand_like(init_image) * 2 - 1
    pertubed_image = init_image.clone().detach()
    # checking value range of pertubed_image
    print(pertubed_image.min(), pertubed_image.max())
    pertubed_image.requires_grad_()
    pbar = tqdm(range(iterations), desc="Iterations")
    success = False
    current_iteration = 0
    # for _, _ in enumerate(pbar):
    while not success and current_iteration < args.max_iter:
        pertubed_image.requires_grad = True
        current_iteration += 1
        id_dist_to_orig = idLossFunc(pertubed_image, init_image)

        # THRESHOLD = args.threshold + args.overhead
        if args.targeted_attack:
            # Calculate the identity loss difference
            tgt_img = Image.open(args.target_image)
            tgt_img = preprocess(tgt_img, res)
            id_dist_to_target = idLossFunc(pertubed_image, tgt_img)

            # THRESHOLD = args.threshold - args.overhead

            attack_loss = -id_dist_to_target + id_dist_to_orig
        else:  # No target attack
            attack_loss = id_dist_to_orig

        # print(id_dist_to_orig)

        loss = args.attack_loss_weight * attack_loss

        loss.backward()

        # PGD: now we update the images
        alpha = args.pgd_alpha
        eps = args.pgd_eps
        with torch.no_grad():
            adv_images = pertubed_image + alpha * pertubed_image.grad.sign()
            eta = torch.clamp(adv_images - init_image, min=-eps, max=+eps)
            pertubed_image = torch.clamp(init_image + eta, min=-1, max=+1).detach_()

        reload_image = tensor2pil2tensor(res, pertubed_image)
        id_distance = frs_model(reload_image, init_image).detach().cpu().numpy()
        # print(id_distance)
        if verbose:
            pbar.set_postfix_str(
                f"iteration: {str(current_iteration)} "
                f"attack_loss: {attack_loss.item():.5f} "
                f"id_loss: {idLossFunc(pertubed_image, init_image):.5f} "
                f"id dist: {id_distance:.5f} "
                f"loss: {loss.item():.5f}"
            )

        if current_iteration > iterations:
            success = True
            # print out attack loss, id_loss and loss
            print(
                f"Attack loss: {attack_loss.item():.5f}, ID loss: {id_distance:.5f}, "
                f"Loss: {loss.item():.5f}"
            )

            break

    """
    Save images
    """

    image = save_results(pertubed_image.detach().cpu(), save_path, init_image)

    blackbox_model_names = [
        key
        for key in threshold_dict
        if key not in args.list_attacker_models and key != args.victim_model
    ]
    thresholds = [threshold_dict[key] for key in blackbox_model_names]
    fr_blackbox_models = [IdLost(name) for name in blackbox_model_names]
    distances_by_blackbox = [
        fr_blackbox_model(reload_image, init_image).detach().cpu().numpy().item()
        for fr_blackbox_model in fr_blackbox_models
    ]
    distance_from_blackbox_dict = dict(zip(blackbox_model_names, distances_by_blackbox))
    recognition_from_blackbox = [
        1 if value > threshold else 0
        for value, threshold in zip(distances_by_blackbox, thresholds)
    ]
    recognition_from_blackbox_dict = dict(
        zip(blackbox_model_names, recognition_from_blackbox)
    )
    image_name = image_path.split("/")[-1]
    log_attack(
        image_name,
        current_iteration,
        idLossFunc(pertubed_image, init_image).detach().cpu().numpy().item(),
        id_distance.item(),
        # id_dist_to_target.detach().cpu().numpy().item(),
        distance_from_blackbox_dict,
        recognition_from_blackbox_dict,
        args,
    )

    return image[0], distance_from_blackbox_dict, recognition_from_blackbox_dict


import torch


def log_attack(
    image_name,
    current_iteration,
    id_loss,
    id_distance,
    distance_from_blackbox_dict,
    recognition_from_blackbox_dict,
    args,
    id_dist_to_target=None,
):
    if id_dist_to_target is not None:
        log_entry = {
            "image_name": image_name,
            "iteration": current_iteration,
            "id_loss": id_loss,
            "ID_victim_distance": id_distance,
            "ID_dist_to_target": id_dist_to_target,
            "distance": distance_from_blackbox_dict,
            "protected": recognition_from_blackbox_dict,
        }
    else:
        log_entry = {
            "image_name": image_name,
            "iteration": current_iteration,
            "id_loss": id_loss,
            "ID_victim_distance": id_distance,
            "distance": distance_from_blackbox_dict,
            "protected": recognition_from_blackbox_dict,
        }

    # Writing to a file
    with open(os.path.join(args.save_dir, "attack_log.json"), "a") as f:
        json.dump(log_entry, f)
        f.write("\n")  # Ensure each log entry is on a new line

    # Using logging
    logging.info(json.dumps(log_entry))
