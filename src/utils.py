import os
from shutil import copy2
from typing import Dict, Optional, Tuple, Union

import timm
import torch
import numpy as np
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from src import Model


def save_checkpoint(state: Dict, is_best: bool, path_state: str, best_state_path: str) -> None:
    """
    :param state: dictionary that includes model parameters to save.
    :param is_best: is this state of the model better than the previous one or not.
    :param path_state: path to save the current state of the model.
    :param best_state_path: path to save the best state of the model.
    """
    torch.save(state, os.path.join(path_state, 'checkpoint.pth'))
    if is_best:
        copy2(os.path.join(path_state, 'checkpoint.pth'), os.path.join(best_state_path, 'model_best.pth'))


def load_check_point(checkpoint_path: str, model, optimizer, device: str):
    """
    checkpoint_path: path to save checkpoint file
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer that we want to load state
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def prepare_results_metrics_print(mean_metrics: Dict, metrics_each_class: Optional[Dict] = None,
                                  json_class: Dict = None) -> Union[Tuple, str]:
    """
    This function makes print metrics more accurate.
    :param mean_metrics: mean value for each metric.
    :param metrics_each_class: mean value for each class according to a certain metric.
    :param json_class: name of classes with id.
    :return: string for print.
    """
    metrics = ''

    for key, value in mean_metrics.items():
        metrics += key + ': ' + '{:4f}'.format(value.item()) + '\n'

    if metrics_each_class is not None:
        metrics_for_each_class = '\n\n'
        for key, value in metrics_each_class.items():
            metrics_for_each_class += '\n' + key + '  _____________________________________\n'
            for i, val_tensor in enumerate(value):
                name = list(json_class.keys())[list(json_class.values()).index(i)]
                metrics_for_each_class += name + ': ' + '{:4f}'.format(val_tensor.item()) + '\n'

        return metrics, metrics_for_each_class

    else:
        return metrics


def get_information_layer(model, type_last_layer: str,
                          image: Optional[torch.Tensor] = None) -> Union[str, Tuple, torch.Tensor]:
    """

    :param model: torch Model
    :param type_last_layer:
    :param image: input torch.Tensor (B, C, W, H)
    :return:
    """
    name_layer = None
    nodes, _ = get_graph_node_names(model)
    for layer in reversed(nodes):
        if type_last_layer in layer:
            name_layer = layer
            break

    if name_layer is None:
        return 'layer not found'

    features = {name_layer: 'out'}
    feature_extractor = create_feature_extractor(model, return_nodes=features)
    return feature_extractor, feature_extractor(image)['out'] if image is not None else feature_extractor


def load_model_weigts(model_name: str, num_classes: int, pretrained: bool, in_chans: int, device: torch.device,
                      weights: Optional[str] = None):
    """
    This function builds a model, you can also load trained weights.
    :param weights: path to .pth file.
    :param model_name: name of the model to build.
    :param num_classes: number of classes.
    :param pretrained: load imagnet weights or not.
    :param in_chans: conversion of input channels to three channels, for training on imagenet scales.
    :param device: place where input data is loaded.
    :return: model
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
    model = Model(model)

    if weights is not None:
        load_parameters = torch.load(weights, map_location=torch.device(device))
        model.load_state_dict(load_parameters['state_dict'])

    model.to(device)

    return model


def tensor_to_numpy_image(tensor_image: torch.Tensor) -> np.array:
    """
    Converting a normalized torch.Tensor image to numpy.array.
    :param tensor_image: input Tensor (B, C, W, H)
    :return: np.array image
    """
    img_tensor = tensor_image[0].permute(1, 2, 0)
    img_tensor = img_tensor * 255.0
    img = np.uint8(img_tensor.to('cpu').numpy())

    return img
