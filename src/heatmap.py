import copy

import PIL.Image
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.cm as mpl_color_map
from PIL import Image
from torchvision import transforms


def apply_colormap_on_image(org_img: PIL.Image.Image, activation: np.array, colormap_name: str = 'hsv'):
    """
    :param org_img: PIL image.
    :param activation: grayscale image.
    :param colormap_name: color model.
    :return: normal image and image with heatmap
    """
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.3
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    heatmap_on_image = Image.new("RGBA", org_img.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_img.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)

    return no_trans_heatmap, heatmap_on_image


def score_cam(input_image: torch.Tensor, conv_output: torch.Tensor, target_class: int, model) -> np.array:
    """
    Score-weighted Class Activation Heatmap based on https://github.com/utkuozbulak
    :param input_image: input Tensor (B, C, W, H).
    :param conv_output: output from last conv layer.
    :param target_class: determining which heatmap class will be output.
    :param model: torch model.
    :return: grayscale image
    """

    cam = np.ones(conv_output[0].shape[1:], dtype=np.float32)
    for i in range(len(conv_output[0])):
        saliency_map = torch.unsqueeze(torch.unsqueeze(conv_output[0][i, :, :], 0), 0)
        saliency_map = F.interpolate(saliency_map, size=(input_image.shape[2], input_image.shape[3]),
                                     mode='bilinear', align_corners=False)
        if saliency_map.max() == saliency_map.min():
            continue
        norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        w = model(input_image*norm_saliency_map)[1][0][target_class]
        cam += w.data.numpy() * conv_output[0][i, :, :].data.numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = np.uint8(cam * 255)
    cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                                                input_image.shape[3]), Image.ANTIALIAS)) / 255
    return cam


def prepare_score_cam(img_tensor: torch.Tensor, conv_output: torch.Tensor, predict:int, model) -> np.array:
    """
    :param img_tensor: input Tensor (B, C, W, H).
    :param conv_output: output from last conv layer.
    :param predict: id of the class that was predicted.
    :param model: torch Model
    :return: np.array image
    """
    transform_to_image = transforms.ToPILImage()
    grayscale_img = score_cam(img_tensor, conv_output, predict, model)
    img_original = transform_to_image(img_tensor[0])

    img_with_heatmap = np.array(apply_colormap_on_image(img_original, grayscale_img)[1])[:, :, :-1]

    return img_with_heatmap
