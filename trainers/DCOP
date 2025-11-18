import os.path as osp
import os
import datetime
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from sklearn.metrics import roc_auc_score

from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn import manifold, datasets
import matplotlib.pyplot as plt
import matplotlib

from augmentation import get_augmenter
from collections import defaultdict

from loss.pair_loss.utils import get_pair_indices
from loss.loss import bha_coeff_loss
from loss.utils.utils import bha_coeff

import torchvision.transforms as T
from torchvision.utils import make_grid
from PIL import Image
import os

_tokenizer = _Tokenizer()



def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, cfg.MODEL.BACKBONE.PATH)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'DCOP',
                      "vision_depth": cfg.TRAINER.DCOP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.DCOP.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.DCOP.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.DCOP.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_ctx = cfg.TRAINER.IVLP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.IVLP.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    @autocast()
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features


@TRAINER_REGISTRY.register()
class IVLP(TrainerXU):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)


        self.prototypes = torch.zeros(self.num_classes, 512).to(self.device)
        self.prototypes_sum = torch.zeros(self.num_classes, 512).to(self.device)
        self.prototypes_count_sum = torch.zeros(self.num_classes, 1).to(self.device)


        self.open_set = cfg.DATASET.OPEN_SET if hasattr(cfg.DATASET, "OPEN_SET") else False
        self.known_classes = cfg.DATASET.KNOWN_CLASSES if hasattr(cfg.DATASET, "KNOWN_CLASSES") else None
        self.unknown_class_id = self.num_classes  
        
        if self.open_set:
            self.thresholds = torch.linspace(0, 1, 100).to(self.device)
            

            if self.known_classes is not None and len(self.known_classes) < self.num_classes:
                self.prototypes = torch.zeros(len(self.known_classes), 512).to(self.device)
                self.prototypes_sum = torch.zeros(len(self.known_classes), 512).to(self.device)
                self.prototypes_count_sum = torch.zeros(len(self.known_classes), 1).to(self.device)

    def check_cfg(self, cfg):
        assert cfg.TRAINER.IVLP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.IVLP.PREC == "fp32" or cfg.TRAINER.IVLP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.n_cls = self.model.prompt_learner.n_cls

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner" 

        for name, param in self.model.named_parameters():
            if name_to_update not in name: 
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set() 
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # transform the epoch to step schedule
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)  
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)  

        self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     # print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model, device_ids=[0])

    def train(self):
        """Generic training loops."""

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset。
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)

        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        # self.test_batches = [int(self.num_batches * 0.33), int(self.num_batches * 0.66)]

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                    self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches

                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                          self.epoch + 1,
                          self.max_epoch,
                          self.batch_idx + 1,
                          self.num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          eta=eta,
                          losses=losses,
                          lr=self.get_current_lr(),
                      ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")


        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        all_image_names = []

        #2025.4.21
        if self.open_set:
            # OpenSet评估需要收集更多信息
            all_labels = []
            all_scores = []
            all_preds = []
            all_max_probs = []

        for batch_idx, batch in enumerate(data_loader):
            input, label, image_paths = self.parse_batch_test(batch)
            output, img_feature = self.model_inference(input)

        
            names = [os.path.basename(p) for p in image_paths]
            all_image_names.extend(names)

            _, preds = torch.max(output, 1)
            correct = (preds == label).squeeze()
            for i in range(len(label)):
                label_id = label[i].item()
                class_total[label_id] += 1
                if correct[i].item():
                    class_correct[label_id] += 1

            # the last second slice is the logits for target domain
            if self.open_set:
                temperature = 1.5  
                probs = F.softmax(output / temperature, dim=1)
                max_probs, preds = torch.max(probs, dim=1)

                all_labels.extend(label.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_max_probs.extend(max_probs.cpu().numpy())

                entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
                unknown_score = entropy / max_probs
                all_scores.extend(unknown_score.cpu().numpy())
                self.evaluator.process(output, label)
            else:
                self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for class_id in sorted(class_total.keys()):
            acc = class_correct[class_id] / class_total[class_id]
            print(f"类别 {class_id}: {acc:.4f} ({class_correct[class_id]}/{class_total[class_id]})")
            results[f"class_{class_id}_accuracy"] = acc

        if self.open_set:
            try:
                from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

                if self.known_classes is not None:
                    binary_labels = np.array([1 if l >= len(self.known_classes) else 0 for l in all_labels])
                else:
                    binary_labels = np.array([1 if l == self.unknown_class_id else 0 for l in all_labels])

                if np.any(binary_labels == 1):  
                    auroc = roc_auc_score(binary_labels, all_scores)
                    results["openset_auroc"] = auroc


                    unknown_threshold = self.cfg.TRAINER.IVLP.UNKNOWN_THRESHOLD if hasattr(self.cfg.TRAINER.IVLP, "UNKNOWN_THRESHOLD") else 0.5
                    predicted_unknown = np.array(all_scores) > unknown_threshold
                    known_indices = np.where(binary_labels == 0)[0]
                    unknown_indices = np.where(binary_labels == 1)[0]

                    unknown_correct = np.sum(predicted_unknown[unknown_indices]) / len(unknown_indices) if len(unknown_indices) > 0 else 0
                    results["unknown_class_accuracy"] = unknown_correct

                   
                    known_class_pred_correct = 0
                    if len(known_indices) > 0:
                       
                        known_and_predicted_known = known_indices[~predicted_unknown[known_indices]]
                        if len(known_and_predicted_known) > 0:
                            
                            true_labels = np.array(all_labels)[known_and_predicted_known]
                            pred_labels = np.array(all_preds)[known_and_predicted_known]
                            known_class_pred_correct = np.sum(true_labels == pred_labels) / len(known_indices)
                    results["known_class_classification_accuracy"] = known_class_pred_correct

                    H_acc = 2 * known_class_pred_correct * unknown_correct /(known_class_pred_correct + unknown_correct)
                    results["H_acc"] = H_acc


                    print(f"OpenSet评估 - AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")
                    print(f"阈值 {unknown_threshold:.2f}")
                    print(f"已知类分类准确率（base acc）: {known_class_pred_correct:.4f}")
                    print(f"未知类分类准确率（novel acc）: {unknown_correct:.4f}")
                    print(f"Harmonic Mean: {H_acc:.4f}")


        
                    if len(known_indices) > 0:
                      
                        known_true_labels = np.array(all_labels)[known_indices]
                        known_pred_labels = np.array(all_preds)[known_indices]
                        known_pure_classification_accuracy = np.mean(known_true_labels == known_pred_labels)
                        results["known_pure_classification_accuracy"] = known_pure_classification_accuracy


                    corrected_preds = np.array(all_preds).copy()
                    corrected_preds[predicted_unknown] = self.unknown_class_id


                    true_labels_with_unknown = np.array(all_labels).copy()
                    if self.known_classes is not None:
                        true_labels_with_unknown[binary_labels == 1] = self.unknown_class_id


                    overall_accuracy_with_unknown = np.mean(corrected_preds == true_labels_with_unknown)
                    results["overall_accuracy_with_unknown"] = overall_accuracy_with_unknown

                
                df = pd.DataFrame({
                    'Image_Name': all_image_names,
                    'True_Labels': all_labels,
                    'Predicted_Labels': all_preds,
                    'Max_Probability': all_max_probs,
                    'Unknown_Score': all_scores,
                    'Is_Unknown': binary_labels
                })
                os.makedirs(self.output_dir, exist_ok=True)
                df.to_csv(osp.join(self.output_dir, 'openset_predictions1.csv'), index=False)

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        if self.open_set:
            results_all = results["H_acc"]
        else:
            results_all = results["accuracy"]

        return results_all



