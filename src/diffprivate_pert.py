import json
import logging
import os
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim
from criteria.id_loss import normalizedIdLoss, sigmoid
from src.utils import tensor2pil2tensor
from src.attCtr import (
    ddim_reverse_sample,
    diffusion_step,
    optimize_unconditional_embeds,
    register_attention_control,
    reset_attention_control,
)

from src.utils import post_process, preprocess, save_results, aggregate_attention
from criteria.lpips.lpips import LPIPS
from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IdLost, EnsembleIdLost

@torch.enable_grad()
def protect(
    model,
    controller,
    args,
    image_path=None,
    save_path="",
):
    # Load image
    image = Image.open(image_path).convert("RGB")
    init_image = preprocess(image, args.diffusion.res)
    height = width = args.diffusion.res

    # Initialize ID loss functions
    if args.model.ensemble:
        idLossFunc = EnsembleIdLost(args.model.list_attacker_models, args.model.ensemble_mode, args)
    else:
        idLossFunc = IdLost(args.model.attacker_model, args)
    frs_model = IdLost(args.model.victim_model, args)

    # Set models to evaluation mode
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    # Prepare prompt
    txt_label = args.attack.object_caption
    prompt = [txt_label] * 2
    true_label = model.tokenizer.encode(txt_label)

    # Access diffusion parameters from args
    num_inference_steps = args.diffusion.diffusion_steps
    guidance_scale = args.diffusion.guidance
    start_step = args.diffusion.start_step
    iterations = args.diffusion.iterations
    verbose = args.diffusion.verbose if hasattr(args.diffusion, 'verbose') else True

    # DDIM sampling
    latent, inversion_latents = ddim_reverse_sample(
        image, prompt, model, num_inference_steps, 0, res=height
    )
    inversion_latents = inversion_latents[::-1]
    latent = inversion_latents[start_step - 1]

    # Optimize unconditional embeddings
    init_prompt = [prompt[0]]
    batch_size = len(init_prompt)
    latent, uncond_embeddings, all_uncond_emb = optimize_unconditional_embeds(
        model,
        guidance_scale,
        start_step,
        width,
        height,
        latent,
        inversion_latents,
        init_prompt,
        batch_size,
    )

    # Attack
    uncond_embeddings.requires_grad_(False)
    register_attention_control(model, controller)
    batch_size = len(prompt)
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = [
        [torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings]
        for i in range(len(all_uncond_emb))
    ]
    context = [torch.cat(i) for i in context]
    original_latent = latent.clone()
    latent.requires_grad_(True)

    # Prepare target image if targeted attack
    if args.attack.targeted_attack:
        tgt_img = Image.open(args.paths.target_image).convert("RGB")
        tgt_img = preprocess(tgt_img, args.diffusion.res)
    else:
        tgt_img = None

    apply_mask = args.diffusion.is_apply_mask
    hard_mask = args.diffusion.is_hard_mask
    init_mask = None if apply_mask else torch.ones([1, 1, *init_image.shape[-2:]]).cuda()

    lr = args.attack.learning_rate
    optimizer = optim.AdamW([latent], lr=lr)
    lpip = LPIPS()
    clip_loss = CLIPLoss()

    pbar = tqdm(range(iterations), desc="Iterations")
    success = False
    current_iteration = 0

    while not success and current_iteration < args.attack.max_iter:
        current_iteration += 1
        controller.loss = 0
        controller.reset()
        latents = torch.cat([original_latent, latent])

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1 :]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

        before_attention_map = aggregate_attention(
            prompt, controller, args.diffusion.res // 32, ("up", "down"), True, 0, is_cpu=False
        )
        after_attention_map = aggregate_attention(
            prompt, controller, args.diffusion.res // 32, ("up", "down"), True, 1, is_cpu=False
        )

        before_true_label_attention_map = before_attention_map[:, :, 1 : len(true_label) - 1]
        after_true_label_attention_map = after_attention_map[:, :, 1 : len(true_label) - 1]

        if init_mask is None:
            init_mask = args.constants.init_mask_scale * torch.nn.functional.interpolate(
                (
                    before_true_label_attention_map.detach().clone().mean(-1)
                    / before_true_label_attention_map.detach().clone().mean(-1).max()
                )
                .unsqueeze(0)
                .unsqueeze(0),
                init_image.shape[-2:],
                mode="bilinear",
            ).clamp(0, 1)
            if hard_mask:
                init_mask = init_mask.gt(args.constants.hard_mask_threshold).float()

        latent_scale = 1 / args.constants.latent_scale
        init_out_image = (
            model.vae.decode(latent_scale * latents)["sample"][1:] * init_mask
            + (1 - init_mask) * init_image
        )

        out_image = post_process(init_out_image)

        # Compute identity distances
        if args.attack.normalize_id_loss:
            id_dist_to_orig = normalizedIdLoss(
                idLossFunc(out_image, init_image),
                threshold=args.attack.threshold + args.attack.overhead,
                apply_sigmoid=args.attack.sigmoid_id,
            )
            THRESHOLD = (
                sigmoid(torch.tensor(1.0)) if args.attack.sigmoid_id else torch.tensor(1.0)
            )
        else:
            id_dist_to_orig = idLossFunc(out_image, init_image)
            id_dist_to_target = (
                idLossFunc(out_image, tgt_img) if tgt_img is not None else None
            )
            THRESHOLD = args.attack.threshold + args.attack.overhead

        # Calculate attack loss
        if args.attack.targeted_attack and tgt_img is not None:
            if args.attack.normalize_id_loss:
                id_dist_to_target = normalizedIdLoss(
                    idLossFunc(out_image, tgt_img),
                    threshold=args.attack.threshold + args.attack.overhead,
                    apply_sigmoid=args.attack.sigmoid_id,
                )

            if args.attack.balance_target:
                identity_diff = id_dist_to_target - id_dist_to_orig
                attack_loss = (
                    torch.abs(identity_diff)
                    if not args.attack.early_penalty or id_dist_to_orig < THRESHOLD
                    else torch.tensor(0.0).cuda()
                )
            else:
                THRESHOLD = args.attack.threshold - args.attack.overhead
                attack_loss = (
                    id_dist_to_target - id_dist_to_orig
                    if not args.attack.early_penalty or id_dist_to_target > THRESHOLD
                    else torch.tensor(0.0).cuda()
                )
        else:
            attack_loss = (
                -id_dist_to_orig
                if not args.attack.early_penalty or id_dist_to_orig < THRESHOLD
                else torch.tensor(0.0).cuda()
            )

        variance_cross_attn_loss = (
            after_true_label_attention_map.var() * args.attack.cross_attn_loss_weight
        )
        self_attn_loss = controller.loss * args.attack.self_attn_loss_weight
        lpips_loss = lpip(out_image, init_image)

        loss = (
            self_attn_loss
            + args.attack.attack_loss_weight * attack_loss
            + variance_cross_attn_loss
            + args.attack.lpips_loss_weight * lpips_loss
        )

        reload_image = tensor2pil2tensor(args.diffusion.res, out_image)
        id_distance = frs_model(reload_image, init_image).detach().cpu().numpy()
        clip_dist = clip_loss.clip_distance(reload_image, init_image).detach().cpu().numpy()

        if verbose:
            pbar.set_postfix_str(
                f"iteration: {current_iteration} "
                f"attack_loss: {attack_loss.item():.5f} "
                f"id_loss: {idLossFunc(out_image, init_image):.5f} "
                f"id dist: {id_distance:.5f} "
                f"clip dist: {clip_dist.item():.5f} "
                f"self_attn_loss: {self_attn_loss.item():.5f} "
                f"loss: {loss.item():.5f}"
            )

        victim_threshold = args.attack.victim_threshold + args.attack.overhead
        if id_distance > victim_threshold:
            success = True
            print(
                f"Attack loss: {attack_loss.item():.5f}, ID loss: {id_distance:.5f}, "
                f"Loss: {loss.item():.5f}"
            )
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save images
    reset_attention_control(model)
    image = save_results(out_image.detach().cpu(), save_path, init_image)

    threshold_dict = args.thresholds

    # Evaluate against blackbox models
    blackbox_model_names = [
        key
        for key in threshold_dict
        if key not in args.model.list_attacker_models and key != args.model.victim_model
    ]
    thresholds = [threshold_dict[key] for key in blackbox_model_names]
    fr_blackbox_models = [IdLost(name, args) for name in blackbox_model_names]
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
    image_name = os.path.basename(image_path)
    log_attack(
        image_name,
        current_iteration,
        idLossFunc(out_image, init_image).detach().cpu().numpy().item(),
        id_distance.item(),
        id_dist_to_target.detach().cpu().numpy().item() if tgt_img is not None else None,
        distance_from_blackbox_dict,
        recognition_from_blackbox_dict,
        args,
    )

    return image[0], distance_from_blackbox_dict, recognition_from_blackbox_dict


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
    with open(os.path.join(args.paths.save_dir, "attack_log.json"), "a") as f:
        json.dump(log_entry, f)
        f.write("\n")  # Ensure each log entry is on a new line

    # Using logging
    logging.info(json.dumps(log_entry))
