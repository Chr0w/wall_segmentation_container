import torch
import torchvision.transforms
from PIL import Image
import numpy as np

from utils.constants import IMAGENET_MEAN, IMAGENET_STD
from utils.utils import IOU, visualize_wall, accuracy
from utils.constants import DEVICE
from tqdm import tqdm


def validation_step(segmentation_module, loader, writer, epoch):
    """
        Function for evaluating the segmentation module on validation dataset
    """
    segmentation_module.eval()
    segmentation_module.to(DEVICE)
    
    total_acc = 0
    total_IOU = 0
    counter = 0
    
    for batch_data in tqdm(loader):
        batch_data = batch_data[0]

        seg_label = np.array(batch_data['seg_label'])
        seg_size = (seg_label.shape[0], seg_label.shape[1])

        with torch.no_grad():
            scores = segmentation_module(batch_data, seg_size=seg_size)

        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()

        # calculate accuracy and IOU
        acc, _ = accuracy(pred, seg_label)
        IOU_curr = IOU(scores.cpu(), seg_label)
        total_IOU += IOU_curr
        total_acc += acc
        counter += 1

    average_acc = total_acc/counter
    average_IOU = total_IOU/counter

    writer.add_scalar('Validation set: accuracy', average_acc, epoch)
    writer.add_scalar('Validation set: IOU', average_IOU, epoch)
    
    return average_acc, average_IOU


def segment_image(segmentation_module, img, disp_image=True, max_size=512):
    """
        Function for segmenting wall in the input image. The input can be path to image, or a loaded image.

        max_size: maximum length of the longest side when running inference. Larger = more VRAM.
                  Use 512 or 768 on GPUs with ~6GB VRAM to avoid OOM. Original resolution mask
                  is produced by resizing the prediction back.
    """
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    if isinstance(img, str):
        img = Image.open(img)

    img_original = np.array(img)
    h_orig, w_orig = img_original.shape[:2]

    # Resize for inference to save VRAM; we'll resize the mask back to original size
    if max_size and (max(h_orig, w_orig) > max_size):
        scale = max_size / max(h_orig, w_orig)
        new_w = int(round(w_orig * scale))
        new_h = int(round(h_orig * scale))
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        img_for_net = pil_to_tensor(img_resized)
        seg_size_infer = (new_h, new_w)
    else:
        img_for_net = pil_to_tensor(img)
        seg_size_infer = (h_orig, w_orig)

    singleton_batch = {'img_data': img_for_net[None].to(DEVICE)}

    with torch.no_grad():
        scores = segmentation_module(singleton_batch, seg_size=seg_size_infer)

    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()

    # Resize prediction back to original image size if we downscaled (nearest-neighbor)
    if max_size and (max(h_orig, w_orig) > max_size):
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
        pred_t = torch.nn.functional.interpolate(
            pred_t, size=(h_orig, w_orig), mode='nearest'
        )
        pred = pred_t[0, 0].numpy().astype(pred.dtype)

    if disp_image:
        visualize_wall(img_original, pred)

    return pred
