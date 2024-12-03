from src.criteria.clip_loss import CLIPLoss


import torch


def clip_distance(x, y):

    pool = torch.nn.AdaptiveAvgPool2d((224, 224))
    # check if x and y have size (224,224), if not, resize
    if x.shape[2] != 224:
        x = pool(x)
    if y.shape[2] != 224:
        y = pool(y)
    clip_feat_x = CLIPLoss().model.encode_image(x)
    clip_feat_y = CLIPLoss().model.encode_image(y)
    sim = torch.nn.functional.cosine_similarity(clip_feat_x, clip_feat_y)

    return 1 - sim.mean()