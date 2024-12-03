import numpy as np
import torch
from torch import nn
import hydra
from omegaconf import DictConfig

from src.models.model_irse import IR_101
import torch.nn.functional as F

import src.models.irse as irse
import src.models.facenet as facenet
import src.models.ir152 as ir152

def distance(embeddings1, embeddings2, distance_metric=1):
    if embeddings1.dim() == 1:
        embeddings1 = embeddings1.unsqueeze(0)
    if embeddings2.dim() == 1:
        embeddings2 = embeddings2.unsqueeze(0)
    if distance_metric == 0:
        # Euclidian distance
        diff = torch.subtract(embeddings1, embeddings2)
        dist = torch.sum(torch.square(diff), dim=1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = torch.sum(torch.multiply(embeddings1, embeddings2), dim=1)
        norm = torch.norm(embeddings1, dim=1) * torch.norm(embeddings2, dim=1)
        similarity = dot / norm
        # Ensure the similarity is within the range [-1, 1] due to floating point arithmetic issues
        # similarity = torch.clamp(similarity, -1.0, 1.0)
        dist = torch.acos(similarity) / torch.pi
    else:
        raise ValueError("Undefined distance metric %d" % distance_metric)
    return dist.squeeze()


class IdLost(nn.Module):
    def __init__(self, model_name, cfg: DictConfig):
        super(IdLost, self).__init__()
        self.model_name = model_name
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fr_model = self.load_fr_model()
        self.pool = torch.nn.AdaptiveAvgPool2d((cfg.id_loss.pool_size, cfg.id_loss.pool_size))
        self.face_pool_112 = torch.nn.AdaptiveAvgPool2d((cfg.id_loss.face_pool_sizes.default, cfg.id_loss.face_pool_sizes.default))
        self.face_pool_160 = torch.nn.AdaptiveAvgPool2d((cfg.id_loss.face_pool_sizes.facenet, cfg.id_loss.face_pool_sizes.facenet))

    def load_fr_model(self):
        fr_model = None
        weights_dir = self.cfg.id_loss.model_weights_dir

        if self.model_name == "ir152":
            fr_model = ir152.IR_152((112, 112))
            fr_model.load_state_dict(
                torch.load(f"{weights_dir}/ir152.pth")
            )
            # print("Loaded IR152 model")
        elif self.model_name == "irse50":
            fr_model = irse.Backbone(50, 0.6, "ir_se")
            fr_model.load_state_dict(
                torch.load(f"{weights_dir}/irse50.pth")
            )
            # print("Loaded IRSE50 model")
        elif self.model_name == "mobile_face":
            fr_model = irse.MobileFaceNet(512)
            fr_model.load_state_dict(
                torch.load(f"{weights_dir}/mobile_face.pth")
            )
            # print("Loaded MobileFace model")
        elif self.model_name == "facenet":
            fr_model = facenet.InceptionResnetV1(num_classes=8631, device=self.device)
            fr_model.load_state_dict(
                torch.load(f"{weights_dir}/facenet.pth")
            )
            # print("Loaded Facenet model")
        elif self.model_name == "cur_face":
            fr_model = IR_101(input_size=112)
            fr_model.load_state_dict(
                torch.load(f"{weights_dir}/CurricularFace_Backbone.pth")
            )
            # print("Loaded CurricularFace model")

        fr_model.to(self.device)
        fr_model.eval()

        return fr_model

    def extract_feats(self, x):
        if x.shape[2] != self.cfg.id_loss.pool_size:
            x = self.pool(x)
        # Crop interesting region using config values
        x = x[:, :, 
            self.cfg.id_loss.face_crop.start_h:self.cfg.id_loss.face_crop.end_h, 
            self.cfg.id_loss.face_crop.start_w:self.cfg.id_loss.face_crop.end_w]
        
        if self.model_name == "facenet":
            x = self.face_pool_160(x)  # convert to 160 x 160
        else:
            x = self.face_pool_112(x)
        x_feats = self.fr_model(x)
        if self.model_name == "ir152":
            x_feats = F.normalize(x_feats, p=None, dim=1)

        return x_feats  # torch.Size([x.shape[0], 512])

    def forward(self, y_hat, y, distance_metric=None):  # y_hat have the gradient
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        
        distance_metric = distance_metric if distance_metric is not None else self.cfg.id_loss.distance_metric
        
        if distance_metric:
            for i in range(n_samples):
                loss += distance(y_hat_feats[i], y_feats[i], distance_metric=self.cfg.id_loss.distance_metric)
                # print(loss)
                count += 1
            return loss / count
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count


class EnsembleIdLost(nn.Module):
    def __init__(self, model_names, mode="mean", cfg=None):
        super(EnsembleIdLost, self).__init__()
        self.models = nn.ModuleList([IdLost(model_name, cfg) for model_name in model_names])
        self.mode = mode
        self.model_names = model_names

    def forward(self, y_hat, y):
        losses = []

        for model, model_name in zip(self.models, self.model_names):
            loss = model(y_hat, y)
            losses.append(loss)

        # Take the average of the features
        if self.mode == "mean":
            avg_loss = torch.mean(torch.stack(losses))
            return avg_loss
        elif self.mode == "max":
            return torch.max(torch.stack(losses))
        elif self.mode == "min":
            return torch.min(torch.stack(losses))
        else:
            # Default to average
            avg_loss = torch.mean(torch.stack(losses))

        return avg_loss


def cosine_similarity(x, y):
    dot_product = torch.dot(x, y)  # Calculate the dot product
    norm_x = torch.norm(x)  # Calculate the norm of x
    norm_y = torch.norm(y)  # Calculate the norm of y
    return dot_product / (norm_x * norm_y)


def Cos_Loss(source_feature, target_feature):
    cos_loss_list = []
    for i in range(len(source_feature)):
        cos_loss_list.append(
            1 - cosine_similarity(source_feature[i], target_feature[i].detach())
        )
        # print(1 - cos_simi(source_feature[i], target_feature[i]))
    cos_loss = torch.mean(torch.stack(cos_loss_list))
    return cos_loss


class EnsembleIdFeats(nn.Module):
    def __init__(self, model_names, cfg=None):
        super(EnsembleIdFeats, self).__init__()
        self.models = nn.ModuleList([IdLost(model_name, cfg) for model_name in model_names])
        self.model_names = model_names

    def extract_feats(self, x):
        features = []

        for model in zip(self.models, self.model_names):
            fr_features = model.extract_feats(x)
            features.append(fr_features)

        # Take the average of the features

        # avg_feat = torch.mean(torch.stack(features))
        return features


def normalizedIdLoss(x, threshold=0.412, apply_sigmoid=True):
    # return sigmoid(x / threshold)
    if apply_sigmoid:
        return torch.sigmoid(x / threshold)
    else:
        return x / threshold


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
