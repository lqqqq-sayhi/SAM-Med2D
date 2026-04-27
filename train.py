import glob
from matplotlib import pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from medpy import metric
import datetime
from torch.nn import functional as F
# from apex import amp
import random

"""
CUDA_VISIBLE_DEVICES=1 nohup python /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/SAM-Med2D/train.py \
--work_dir /mnt/hdd2/task2/sam-med2d \
--run_name exp_6_fold0_resume \
--epochs 200 \
--data_path /mnt/hdd2/task2/sam_lora/train \
--data_path_val /mnt/hdd2/task2/sam_lora/val \
--train_mode train1 \
--val_mode val1 \
> /mnt/hdd2/task2/sam-med2d/exp_6_fold0_resume.log 2>&1 &

如果要resume训练，添加参数：
--resume /mnt/hdd2/task2/sam-med2d/sam-med2d_b.pth \
> /mnt/hdd2/task2/sam-med2d/exp_6_fold0_resume.log 2>&1 &
"""

def parse_args():
    """
    work_dir: Specifies the working directory for the training process. Default value is workdir.
    image_size: Default value is 256.
    mask_num: Specify the number of masks corresponding to one image, with a default value of 5.
    data_path: Dataset directory, for example: data_demo.
    resume: Pretrained weight file, ignore sam_checkpoint if present.
    sam_checkpoint: Load sam checkpoint.
    iter_point: Mask decoder iterative runs.
    multimask: Determines whether to output multiple masks. Default value is True.
    encoder_adapter: Whether to fine-tune the Adapter layer, set to False only for fine-tuning the decoder.
    use_amp: Set whether to use mixed-precision training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sam-med2d", help="run model name")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path") 
    parser.add_argument("--data_path_val", type=str, default="data_demo", help="validation data path")
    parser.add_argument("--train_mode", type=str, default="train", help="training mode")
    parser.add_argument("--val_mode", type=str, default="val", help="validation mode")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="/mnt/hdd2/task2/sam-med2d/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=False, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=False, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args

def calculate_iou(pred, target):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Arguments:
        pred: Predicted mask (tensor)
        target: Ground truth mask (tensor)
    
    Returns:
        iou: IoU score (tensor scalar)
        hd95
    """
    pred_binary = (pred > 0.5).float()
    label_binary = (target > 0.5).float()
    pred_binary = pred_binary.cpu().numpy().astype(bool)
    label_binary = label_binary.cpu().numpy().astype(bool)

    intersection = np.logical_and(pred_binary, label_binary)
    union = np.logical_or(pred_binary, label_binary)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    
    return iou

def calculate_metrics(pred, target):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Arguments:
        pred: Predicted mask (tensor)
        target: Ground truth mask (tensor)
    
    Returns:
        iou: IoU score (tensor scalar)
        hd95
    """
    pred_binary = (pred > 0.5).float()
    label_binary = (target > 0.5).float()
    pred_binary = pred_binary.cpu().numpy().astype(bool)
    label_binary = label_binary.cpu().numpy().astype(bool)

    intersection = np.logical_and(pred_binary, label_binary)
    union = np.logical_or(pred_binary, label_binary)
    dice = (2.0 * np.sum(intersection)) / (np.sum(pred_binary) + np.sum(label_binary) + 1e-8)

    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    
    try:
        if np.sum(pred_binary) > 0 and np.sum(label_binary) > 0:
            hd95 = metric.binary.hd95(pred_binary, label_binary)
        else:
            hd95 = np.nan
    except:
        hd95 = np.nan

    return dice, iou, hd95

def manage_checkpoints(save_dir, max_keep=3):
    """保留最近max_keep个检查点，删除更旧的"""
    # 获取所有检查点并按修改时间排序
    all_checkpoints = glob.glob(os.path.join(save_dir, "*.pth"))
    all_checkpoints.sort(key=os.path.getmtime)
    
    # 删除旧检查点（保留最新的max_keep个）
    if len(all_checkpoints) > max_keep:
        for old_checkpoint in all_checkpoints[:-max_keep]:
            try:
                # 跳过epochX_sam_best.pth
                if os.path.basename(old_checkpoint).replace(".pth", "").split("_")[-1] == "best":
                    print(f"best model {os.path.basename(old_checkpoint)}, 不删除")
                else:
                    os.remove(old_checkpoint)
                    print(f"删除旧检查点: {os.path.basename(old_checkpoint)}")
            except Exception as e:
                print(f"删除失败 {old_checkpoint}: {str(e)}")
                
def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )
  
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    """
    For each image–mask pair:
    Iteration 0: Decoder predicts a mask from an initial prompt (e.g. point/box).
    Iteration 1+:
    Compare predicted mask vs GT mask for this specific label.
    Use generate_point() to add new corrective prompts at error regions.
    Run decoder again (same image embedding, new prompts).

    Each GT mask is treated separately.
    Each mask goes through several rounds of predict → error clicks → refine.
    """
    train_loader = tqdm(train_loader)
    train_losses = []
    train_ious = []
    train_iter_metrics = [0] * len(args.metrics)
    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        # batched_image["image"].shape: [batch_size, 1, 3, 256, 256]
        # batched_image["label"].shape: [batch_size, mask_num, 1, 256, 256]
        
        if random.random() > 0.5:
            batched_input["point_coords"] = None
            flag = "boxes"
        else:
            batched_input["boxes"] = None
            flag = "point"

        for n, value in model.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        # labels is a stacked tensor of masks: [num_classes_for_this_img, H, W]
        if args.use_amp:
            labels = batched_input["label"].half()
            image_embeddings = model.image_encoder(batched_input["image"].half())
  
            B, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            for i in range(B):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False)
            loss = criterion(masks, labels, iou_predictions)
            # use_amp = False so I comment:
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward(retain_graph=False)

        else:
            labels = batched_input["label"]
            image_embeddings = model.image_encoder(batched_input["image"])

            B, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            for i in range(B):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False)
            loss = criterion(masks, labels, iou_predictions)
            loss.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()

        if int(batch+1) % 50 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch+1}, first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}')

        # Finds error regions (prediction ≠ GT) using select_random_points()
        # samples point_num coordinates -> batched_input as new prompts
        point_num = random.choice(args.point_list)
        batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
        batched_input = to_device(batched_input, args.device)
    
        image_embeddings = image_embeddings.detach().clone()
        for n, value in model.named_parameters():
            if "image_encoder" in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        # Refinement loop
        # The final mask is the result of this iterative refinement process
        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for iter in range(args.iter_point):
            if iter == init_mask_num or iter == args.iter_point - 1:
                # the next decoder iteration only uses the image embeddings and not any past prompt history.
                batched_input = setting_prompt_none(batched_input)

            if args.use_amp:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)
                loss = criterion(masks, labels, iou_predictions)
                # use_amp = False so I comment:
                # with amp.scale_loss(loss,  optimizer) as scaled_loss:
                #     scaled_loss.backward(retain_graph=True)
            else:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)
                loss = criterion(masks, labels, iou_predictions)
                loss.backward(retain_graph=True)
                
            optimizer.step()
            optimizer.zero_grad()
          
            if iter != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                batched_input = to_device(batched_input, args.device)
       
            if int(batch+1) % 50 == 0:
                if iter == init_mask_num or iter == args.iter_point - 1:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
                else:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: { SegMetrics(masks, labels, args.metrics)}')

        if int(batch+1) % 200 == 0:
            print(f"epoch:{epoch+1}, iteration:{batch+1}, loss:{loss.item()}")
            save_dir = os.path.join(f"{args.work_dir}/models", args.run_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"epoch{epoch+1}_batch{batch+1}_sam.pth")
            state = {'model': model.state_dict(), 'optimizer': optimizer}
            try:
                torch.save(state, save_path)
                print(f"模型成功保存到 {save_path}")
                manage_checkpoints(save_dir, max_keep=5)
            except Exception as e:
                print(f"保存失败: {e}")
                

        # refinement metrics: how the segmentation improves as corrective points are added.
        train_losses.append(loss.item())

        batch_iou = calculate_iou(masks, labels)
        train_ious.append(batch_iou)



        gpu_info = {}
        gpu_info['gpu_name'] = args.device 
        train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

        train_batch_metrics = SegMetrics(masks, labels, args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics, train_ious

# Validation, can choose refinement or not
def validate_one_epoch(args, model, val_loader, criterion):
    # "cold": no refinement
    # "refine": iterative refinement
    validate_mode = "cold"
    model.eval()
    val_losses, val_dices, val_ious, val_hd95s = [], [], [], []
    with torch.no_grad():
        for batch, batched_input in enumerate(tqdm(val_loader, desc="Validating")):
            # prepare batch
            batched_input = stack_dict_batched(batched_input)
            batched_input = {k: (v.float().to(args.device) if isinstance(v, torch.Tensor) else v) 
                             for k, v in batched_input.items()}

            # choose point vs box prompt randomly
            if random.random() > 0.5:
                batched_input["point_coords"] = None
                flag = "boxes"
            else:
                batched_input["boxes"] = None
                flag = "point"

            # encode image
            if args.use_amp:
                labels = batched_input["label"].half()
                image_embeddings = model.image_encoder(batched_input["image"].half())
            else:
                labels = batched_input["label"]
                image_embeddings = model.image_encoder(batched_input["image"])

            # repeat embedding per mask
            B, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            for i in range(B):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            # --------------------------
            # MODE 1: Cold (no refinement)
            # --------------------------
            if validate_mode == "cold":
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings, decoder_iter=False
                )
                loss = criterion(masks, labels, iou_predictions)
                val_losses.append(loss.item())

                dice, iou, hd95 = calculate_metrics(masks, labels)
                val_dices.append(dice)
                val_ious.append(iou)
                val_hd95s.append(hd95)

            # --------------------------
            # MODE 2: Refine (iterative with corrective points)
            # --------------------------
            elif validate_mode == "refine":
                # initial forward
                masks, low_res_masks, iou_predictions = prompt_and_decoder(
                    args, batched_input, model, image_embeddings, decoder_iter=False
                )
                loss = criterion(masks, labels, iou_predictions)
                val_losses.append(loss.item())
                dice, iou, hd95 = calculate_metrics(masks, labels)
                val_dices.append(dice)
                val_ious.append(iou)
                val_hd95s.append(hd95)

                # iterative refinement loop
                # remove generate_point() since validation need to reflect how well the model generalizes without guidance.
                init_mask_num = np.random.randint(1, args.iter_point - 1)
                for iter in range(args.iter_point):
                    if iter == init_mask_num or iter == args.iter_point - 1:
                        batched_input = setting_prompt_none(batched_input)

                    masks, low_res_masks, iou_predictions = prompt_and_decoder(
                        args, batched_input, model, image_embeddings, decoder_iter=False
                    )

                    loss = criterion(masks, labels, iou_predictions)
                    val_losses.append(loss.item())

                    dice, iou, hd95 = calculate_metrics(masks, labels)
                    val_dices.append(dice)
                    val_ious.append(iou)
                    val_hd95s.append(hd95)

                    if iter != args.iter_point - 1:
                        point_num = random.choice(args.point_list)
                        batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)

            else:
                raise ValueError(f"Unknown validate_mode: {validate_mode}. Choose 'cold' or 'refine'.")

    # clean up hd95
    valid_hd95s = [x for x in val_hd95s if not np.isnan(x)]
    return (np.mean(val_losses), np.mean(val_dices), np.mean(val_ious), 
            (np.mean(valid_hd95s) if valid_hd95s else np.nan))

def main(args):
    model = sam_model_registry[args.model_type](args).to(args.device) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.5)
        print('*******Use MultiStepLR')

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            print(f"*******load {args.resume}")

    if args.use_amp:
        # use_amp = False so I comment:
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print("*******Mixed precision with Apex")
    else:
        print('*******Do not use mixed precision')

    train_dataset = TrainingDataset(args.data_path, 
                                    image_size=args.image_size, 
                                    mode=args.train_mode, 
                                    point_num=1, 
                                    mask_num=args.mask_num, 
                                    requires_name = False)
    train_loader = DataLoader(train_dataset, 
                              batch_size = args.batch_size, 
                              shuffle=True, num_workers=4)
    print('*******Train data:', len(train_dataset))   

    val_dataset = TrainingDataset(args.data_path_val, 
                                  image_size=args.image_size, 
                                  mode=args.val_mode, 
                                  point_num=1, 
                                  mask_num=args.mask_num, requires_name=False)
    val_loader = DataLoader(val_dataset, 
                            batch_size=1, 
                            shuffle=False, num_workers=2)
    print('*******Val data:', len(val_dataset))


    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
    print(f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}")
    best_loss = 1e10
    l = len(train_loader)

    train_loss_history, val_loss_history = [], []
    train_macro_iou_history = []
    train_iou_history = []
    val_dice_history, val_iou_history, val_hd95_history = [], [], []

    best_val_iou = 0.0
    patience, no_improve_epochs = 5, 0
    lr_reduce_count, max_lr_reduces = 0, 5

    for epoch in range(0, args.epochs):
        model.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        train_losses, train_iter_metrics, train_ious = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion)
        val_loss, val_dice, val_iou, val_hd95 = validate_one_epoch(args, model, val_loader, criterion)

        if args.lr_scheduler is not None:
            scheduler.step()

        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

        average_loss = np.mean(train_losses)
        train_loss_history.append(average_loss)

        train_macro_iou_history.append(train_metrics['iou'])

        mean_train_iou = np.mean(train_ious)
        train_iou_history.append(mean_train_iou)

        val_loss_history.append(val_loss)
        val_dice_history.append(val_dice)
        val_iou_history.append(val_iou)
        val_hd95_history.append(val_hd95)

        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr

        # Macro-average IoU: 先为每个样本计算IoU，再对所有样本的IoU求平均
        # metrics: {'iou': '0.6318'} → Average of per-mask IoUs, then divided samples of this batch, then divided nums of batch in current epoch.
        # Averaging per-sample IoUs tends to be higher (each image with small objects gets fair weight).
        
        # Micro-average IoU:
        # Train IoU: 0.4598 → Global IoU across all images in current epoch (intersection and union summed first, then divided).
        # Global IoU tends to be lower (tiny structures are overwhelmed by large background unions).
        loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}, Train IoU: {mean_train_iou:.4f}")
        print(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}, Train IoU: {mean_train_iou:.4f}")
        
        loggers.info(f"epoch: {epoch + 1}, Val loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val HD95: {val_hd95:.4f}")
        print(f"epoch: {epoch + 1}, Val loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val HD95: {val_hd95:.4f}")
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            no_improve_epochs = 0
            save_dir = os.path.join(args.work_dir, "models", args.run_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"epoch{epoch+1}_sam_best_val.pth")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}
            try:
                torch.save(state, save_path)
                print(f"模型成功保存到 {save_path}")
                # 删除旧的最佳模型（如果存在）
                save_path_old = os.path.join(save_dir, f"epoch{epoch}_sam_best_val.pth")
                if os.path.exists(save_path_old):
                    try:
                        os.remove(save_path_old)
                        print(f"删除旧最佳模型")
                    except Exception as e:
                        print(f"删除旧最佳模型失败: {str(e)}")
            except Exception as e:
                print(f"保存失败: {e}")
                
                
        else:
            no_improve_epochs += 1
            print(f"No improvement in IoU for {no_improve_epochs}/{patience} epochs")
            loggers.info(f"No improvement in IoU for {no_improve_epochs}/{patience} epochs")

        if no_improve_epochs >= patience:
            if lr_reduce_count < max_lr_reduces:
                    # reduce LR by factor of 10
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10.0
                    lr_reduce_count += 1
                    print(f"Reducing LR by 10x → new LR: {optimizer.param_groups[0]['lr']:.1e} "
                          f"(total reduces {lr_reduce_count}/{max_lr_reduces})")
                    no_improve_epochs = 0  # reset counter
            else:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    loggers.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
        if average_loss < best_loss:
            best_loss = average_loss
            save_dir = os.path.join(args.work_dir, "models", args.run_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"epoch{epoch+1}_sam_best.pth")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}
            try:
                torch.save(state, save_path)
                print(f"模型成功保存到 {save_path}")
                # 删除旧的最佳模型（如果存在）
                save_path_old = os.path.join(save_dir, f"epoch{epoch}_sam_best.pth")
                if os.path.exists(save_path_old):
                    try:
                        os.remove(save_path_old)
                        print(f"删除旧最佳模型")
                    except Exception as e:
                        print(f"删除旧最佳模型失败: {str(e)}")            
            except Exception as e:
                print(f"保存失败: {e}")
            # use_amp = False so I comment:
            # if args.use_amp:
            #     model = model.half()

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))
        loggers.info("Run epoch time: %.2fs" % (end - start))

    plt.figure(figsize=(15, 12))

    plt.subplot(2, 2, 1)
    plt.plot(train_loss_history, label='Training Mean Loss', marker='o')
    plt.plot(val_loss_history, label='Validation Mean Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Mean Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_iou_history, label='Training Mean IoU', color='orange', marker='o')
    plt.plot(train_macro_iou_history, label='Training Macro IoU', color='green', marker='^')
    plt.plot(val_iou_history, label='Validation Mean IoU', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Validation Mean IoU')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(val_dice_history, label='Validation Mean Dice', marker='s')
    plt.plot(val_iou_history, label='Validation Mean IoU', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Mean Dice and Mean IoU')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(val_hd95_history, label='Validation Mean HD95', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('HD95')
    plt.title('Validation Mean HD95')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(args.work_dir, 
                             "plots", 
                             f"train_val_metrics_{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.png")    
    plt.savefig(plot_path)
    print(f"Training and validation metrics visualization saved to {plot_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)


