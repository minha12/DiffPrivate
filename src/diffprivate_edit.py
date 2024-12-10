import argparse
import json
import logging
import math
import os

# from torch.cuda import amp
import torch

# import torchvision
from torch import optim

# import torch.nn.functional as F
from tqdm import tqdm

# from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IdLost as IDLoss
from criteria.lpips.lpips import LPIPS
# from criteria.id_loss import threshold_dict

# from dataset import CelebAttrDataset, ImageDataset
from diffae_cls import ClsModel

# from model import Model
from diffae import LitModel
from general_config import ffhq256_autoenc
from general_config import ffhq256_autoenc_cls
from src.utils import save_edited_results, tensor2numpy, tensor2pil2tensor

threshold_dict = {
    "irse50": 0.4,
    "ir152": 0.4277,
    "facenet": 0.33519999999999994,
    "cur_face": 0.4332,
    "mobile_face": 0.3875,
}

device = "cuda:0"
conf = ffhq256_autoenc()
model_resolution = 256
classifer_config = ffhq256_autoenc_cls()
weight_dir_path = f"src/DiffAE/checkpoints/{classifer_config.name}"


def _load_diffae():
    # load diffae
    model = LitModel(conf)
    model.ema_model.requires_grad_(False)
    model.model.requires_grad_(False)
    state = torch.load(f"src/DiffAE/checkpoints/{conf.name}/last.ckpt", map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)
    # model.model.to(device)

    return model


def _load_cls_model():
    # load semantic model
    cls_conf = ffhq256_autoenc_cls()
    cls_model = ClsModel(cls_conf)
    state = torch.load(f"src/DiffAE/checkpoints/{cls_conf.name}/last.ckpt", map_location="cpu")
    print("latent step:", state["global_step"])
    cls_model.load_state_dict(state["state_dict"], strict=False)
    cls_model.requires_grad_(False)
    cls_model.eval()
    cls_model.to(device)
    return cls_model


def convert2rgb(img, adjust_scale=True):
    convert_img = torch.tensor(img)
    if adjust_scale:
        convert_img = (convert_img + 1) / 2
    return convert_img.cpu()


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def attack_sem(
    args, model, cond, xT, cond_new, src_img, tar_img, save_path, verbo=True
):
    # print(args)
    os.makedirs(args.results_dir, exist_ok=True)
    id_loss = IDLoss(args.attacker_model)

    beta = torch.zeros_like(cond).clone().cuda()

    beta.requires_grad = True
    print(beta.dtype)
    lpips_loss = LPIPS(net_type="alex").to("cuda:0").eval()

    optimizer = optim.Adam([beta], lr=0.1)

    print("optimizing ...")
    if verbo:
        pbar = tqdm(range(args.max_iter))
    else:
        pbar = range(args.max_iter)
    current_iteration = 0
    success = False
    while not success and current_iteration < args.max_iter:
        current_iteration += 1
        t = current_iteration / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        cond2 = beta * cond + (1 - beta) * cond_new

        # with torch.cuda.amp.autocast():
        gen_img = model.render(xT.detach(), cond2, T=7)

        loss_lpips = lpips_loss(gen_img, src_img.cuda())
        diff_src = id_loss(gen_img, src_img.cuda())
        if args.targeted_attack:
            diff_tar = id_loss(gen_img, tar_img.cuda())

            i_loss = diff_tar - diff_src
            # loss = 1 - diff

        else:
            i_loss = -diff_src

        l2_loss = ((cond2 - cond.cuda()) ** 2).sum()
        loss = (
            args.lpips_lambda * loss_lpips
            + args.id_lambda * i_loss
            + args.l2_lambda * l2_loss
        )
        reload_image = tensor2pil2tensor(args.img_size, gen_img)
        frs_model = IDLoss(args.victim_model)
        diff = frs_model(reload_image, src_img)
        vimtim_threshold = args.victim_threshold + args.overhead
        if diff > vimtim_threshold:
            success = True
            # print out attack loss, id_loss and loss
            print(
                f"ID loss: {diff_src.item():.5f}, ID Dist: {diff:.5f}, "
                f"Loss: {loss.item():.5f}"
            )
            # get image name form image_path

            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbo:
            pbar.set_description(
                (
                    f"iteration: {current_iteration} --- lpips loss: {args.lpips_lambda * loss_lpips.item():.4f} --- id_loss: {diff_src:.4f} -- l2 loss: {args.l2_lambda * l2_loss.item():.4f} -- diff: {diff:.4f} --diff_tar: {diff_tar:.4f}"
                )
            )

    blackbox_model_names = [
        key
        for key in threshold_dict
        if key != args.attacker_model and key != args.victim_model
    ]
    thresholds = [threshold_dict[key] for key in blackbox_model_names]
    fr_blackbox_models = [IDLoss(name) for name in blackbox_model_names]
    distances_by_blackbox = [
        fr_blackbox_model(reload_image, src_img.cuda()).detach().cpu().numpy().item()
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
    image_name = save_path.split("/")[-1]
    log_attack(
        image_name,
        current_iteration,
        id_loss(gen_img, src_img.cuda()).detach().cpu().numpy().item(),
        diff.item(),
        diff_tar.detach().cpu().numpy().item(),
        distance_from_blackbox_dict,
        recognition_from_blackbox_dict,
        args,
    )

    save_edited_results(
        tensor2numpy(gen_img),
        model_name="diffae",
        save_path=save_path,
        init_image=src_img,
    )
    return gen_img


def log_attack(
    image_name,
    current_iteration,
    id_loss,
    id_distance,
    id_dist_to_target,
    distance_from_blackbox_dict,
    recognition_from_blackbox_dict,
    args,
):
    log_entry = {
        "image_name": image_name,
        "iteration": current_iteration,
        "id_loss": id_loss,
        "ID_victim_distance": id_distance,
        "ID_dist_to_target": id_dist_to_target,
        "distance": distance_from_blackbox_dict,
        "protected": recognition_from_blackbox_dict,
    }

    # Writing to a file
    log_dir = os.path.join(args.results_dir, "logs")
    # check if log directory exists, if not create it
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "attack_log.json"), "a") as f:
        json.dump(log_entry, f)
        f.write("\n")  # Ensure each log entry is on a new line

    # Using logging
    logging.info(json.dumps(log_entry))


def project(zadv, z, gamma):
    diff = zadv - z
    clamped_diff = diff.clamp(-gamma, gamma)
    projected_zadv = z + clamped_diff
    return projected_zadv


def attack(args, model, cond, xT, src_img, tar_img, save_path, verbo=True):
    os.makedirs(args.results_dir, exist_ok=True)
    id_loss = IDLoss(args.attacker_model)
    frs_model = IDLoss(args.victim_model)
    # create z_adv zero like cond
    # z_adv = torch.zeros_like(cond).clone().cuda()
    z_adv = cond.clone().cuda()

    z_adv.requires_grad = True
    print(z_adv.dtype)
    lpips_loss = LPIPS(net_type="alex").to("cuda:0").eval()

    optimizer = optim.Adam([z_adv], lr=args.lr)

    print("optimizing ...")
    if verbo:
        pbar = tqdm(range(args.max_iter))
    else:
        pbar = range(args.max_iter)
    current_iteration = 0
    success = False
    while not success and current_iteration < args.max_iter:
        current_iteration += 1
        t = current_iteration / args.max_iter
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        # cond2 = beta * cond + (1 - beta) * cond_new

        # with torch.cuda.amp.autocast():
        gen_img = model.render(xT.detach(), z_adv, T=7)

        loss_lpips = lpips_loss(gen_img, src_img.cuda())
        diff_src = id_loss(gen_img, src_img.cuda())
        if args.targeted_attack:
            diff_tar = id_loss(gen_img, tar_img.cuda())
            # diff_src = id_loss(gen_img, src_img.cuda())
            i_loss = diff_tar - diff_src
            # loss = 1 - diff

        else:
            i_loss = -diff_src

        l2_loss = ((z_adv - cond.cuda()) ** 2).sum()

        loss = (
            args.lpips_lambda * loss_lpips
            + args.id_lambda * i_loss
            + args.l2_lambda * l2_loss
        )
        vimtim_threshold = threshold_dict[args.victim_model] + args.overhead
        reload_image = tensor2pil2tensor(args.img_size, gen_img)

        diff = frs_model(reload_image, src_img)
        vimtim_threshold = threshold_dict[args.victim_model] + args.overhead
        if diff > vimtim_threshold:
            success = True
            # print out attack loss, id_loss and loss
            print(
                f"ID loss: {diff_src.item():.5f}, ID Dist: {diff:.5f}, "
                f"Loss: {loss.item():.5f}"
            )
            # get image name form image_path

            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbo:
            pbar.set_description(
                (
                    f"iteration: {current_iteration} --- lpips loss: {args.lpips_lambda * loss_lpips.item():.4f} --- id_loss: {diff_src:.4f} -- l2 loss: {args.l2_lambda * l2_loss.item():.4f} -- diff: {diff:.4f}"
                )
            )
    save_edited_results(
        tensor2numpy(gen_img),
        model_name="diffae",
        save_path=save_path,
        init_image=src_img,
    )
    return gen_img


def diffprotect_v2(args, model, cond, xT, src_img, tar_img, verbo=True):
    print(args)
    os.makedirs(args.results_dir, exist_ok=True)
    eta = 2 * args.lamba / args.step
    id_loss = IDLoss()
    z_adv = cond.clone().cuda()

    z_adv.requires_grad = True

    lpips_loss = LPIPS(net_type="alex").to("cuda:0").eval()

    optimizer = optim.Adam([z_adv], args.lr)

    print("optimizing ...")
    if verbo:
        pbar = tqdm(range(args.step))
    else:
        pbar = range(args.step)

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        gen_img = model.render(xT.detach(), z_adv, T=7)

        loss_lpips = lpips_loss(gen_img, src_img.cuda())

        if args.id_lambda > 0:
            i_loss = id_loss(gen_img, tar_img[None][0][0].cuda().unsqueeze(0))[0]
            # loss = 1 - diff
        else:
            i_loss = 0

        l2_loss = ((gen_img - src_img.cuda()) ** 2).sum()
        total_loss = (
            args.lpips_lambda * loss_lpips
            + args.id_lambda * i_loss
            + args.l2_lambda * l2_loss
        )

        optimizer.zero_grad()
        total_loss.backward()

        # Gradient based step, followed by projection
        z_adv_new = z_adv - eta * torch.sign(z_adv.grad.data)
        z_adv_new = project(z_adv_new, cond, args.lamba)
        # Reset gradients for next step
        if z_adv.grad is not None:
            z_adv.grad.zero_()

        z_adv = z_adv_new.detach()
        z_adv.requires_grad = True

        if verbo:
            pbar.set_description(
                (f"loss: {total_loss.item():.4f} --- id_loss: {i_loss:.4f}")
            )
    with torch.no_grad():
        gen_img = model.render(xT.detach(), z_adv, T=7)
    return gen_img
