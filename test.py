import os
import sys
import json
from typing import Dict, Tuple, Union

import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from memory_profiler import profile


from src import (CustomDataset, MetricsTorch, prepare_results_metrics_print, get_information_layer,
                 load_model_weigts, tensor_to_numpy_image, prepare_score_cam)
from config import (PATH_DATA, JSON_NAME, CLASS_JSON, AUGMENTATION_DATA, MODEL_NAME, INPUT_SHAPE_IMAGE, GPU_NUM,
                    NUMBER_CLASSES)


@profile(precision=4)
def count_metrics(weights: str, json_file: str = JSON_NAME, json_class: str = CLASS_JSON, data_path: str = PATH_DATA):

    with open(os.path.join(data_path, json_class), 'r') as file:
        data_name = json.load(file)

    test_data = CustomDataset(data_path=data_path, json_name=json_file, is_train='test',
                              image_shape=INPUT_SHAPE_IMAGE, augmentation_data=False)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4,
                             persistent_workers=True)

    device = torch.device('cuda:{}'.format(GPU_NUM) if torch.cuda.is_available() else 'cpu')

    model = load_model_weigts(weights=weights, model_name=MODEL_NAME, num_classes=NUMBER_CLASSES, pretrained=False,
                              in_chans=INPUT_SHAPE_IMAGE[0], device=device)
    model.eval()

    metric = MetricsTorch(average='macro', device=device, accuracy=True, precision=True, recall=True, f1=True,
                          each_class=True)

    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        _, soft_out = model(inputs)
        _, preds = torch.max(soft_out, 1)
        metric.validation_step(labels, preds)
        metric.validation_step_each_class(labels, preds)

    val_epoch_metrics = metric.validation_step_compute()
    val_epoch_metrics_each_class = metric.validation_step_each_class_compute()

    general_metric, metric_each_class = prepare_results_metrics_print(val_epoch_metrics,
                                                                      val_epoch_metrics_each_class, data_name)
    print(general_metric)
    print(metric_each_class)


def show_batch(data_path: str = PATH_DATA, json_name: str = JSON_NAME, class_json: str = CLASS_JSON,
               is_train: bool = True, augmentation_data: bool = AUGMENTATION_DATA):
    """
    This function shows image from data_generator.py.
    :param data_path: path for data.
    :param json_name: the name of the json file that contains information about the files to download.
    :param class_json: json file that contains information about the index and class name.
    :param is_train: if is_train = True, then we work with train images, otherwise with test.
    :param augmentation_data: if this parameter is True, then augmentation is applied to the training dataset.
    """

    # load json with information index and class.
    with open(os.path.join(data_path, class_json), 'r') as file:
        class_name = json.load(file)

    data_show = CustomDataset(data_path=data_path, json_name=json_name, is_train='train', shuffle_data=True)

    for i in range(len(data_show)):
        image, label = data_show[i]
        image = tensor_to_numpy_image(torch.unsqueeze(image, 0))
        name = list(class_name.keys())[list(class_name.values()).index(label.numpy())]

        if is_train is True and augmentation_data is True:
            augmentation = True
        else:
            augmentation = False

        name_window = str(name) + ' ' + 'аугментация = ' + str(augmentation)

        plt.imshow(image)
        plt.title(name_window)
        if plt.waitforbuttonpress(0):
            plt.close('all')
            return
        plt.close('all')


def show(path_file: str, label, probability, image_heatmap=None) -> None:
    """
    Show image or images using matplotlib.
    :param path_file: pth for original file.
    :param label: image class
    :param probability: probability of belonging to a class.
    :param image_heatmap: image with heatmap.
    """
    img = cv2.cvtColor(cv2.imread(path_file), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(9, 9))

    if image_heatmap is not None:
        plt.subplot(2, 1, 1)
        plt.imshow(image_heatmap)
        plt.title('SCORE_CAM')

        plt.subplot(2, 1, 2)
        plt.imshow(img)
        plt.title(label + ' - probability: {:.3f}'.format(probability))

    else:
        plt.imshow(img)
        plt.title(label + ' - probability: {:.3f}'.format(probability))

    if plt.waitforbuttonpress(0):
        plt.close('all')
        sys.exit()
    plt.close('all')


def show_image_prediction(path_file_dir: str, weights: str, json_class: str, apply_score_cam: bool = False) -> None:
    """
    Show an image with predict name and probability. You can also optionally display a heatmap.
    :param path_file_dir: path to a file or a folder that contains images.
    :param weights: path to .pth file.
    :param json_class: path to json file that contains names and ID's.
    :param apply_score_cam: making a heatmap based on score_cam.
    """

    with open(json_class, 'r') as file:
        id_class = json.load(file)

    device = torch.device('cuda:{}'.format(GPU_NUM) if torch.cuda.is_available() else 'cpu')
    model = load_model_weigts(weights=weights, model_name=MODEL_NAME, num_classes=NUMBER_CLASSES, pretrained=False,
                              in_chans=INPUT_SHAPE_IMAGE[0], device=device)
    model.eval()

    prepare_data = CustomDataset(is_train='val')

    if os.path.isdir(path_file_dir):
        for file in os.listdir(path_file_dir):
            full_path = os.path.join(path_file_dir, file)
            prediction, label, img, conv, probability = predict(full_path, model, prepare_data,
                                                                id_class, conv_output=apply_score_cam)
            if apply_score_cam:
                image_heatmap = prepare_score_cam(img, conv, prediction, model)
                show(full_path, label, probability, image_heatmap)
            else:
                show(full_path, label, probability)
    else:
        prediction, label, img, conv, probability = predict(path_file_dir, model, prepare_data,
                                                            id_class, conv_output=apply_score_cam)

        if apply_score_cam:
            image_heatmap = prepare_score_cam(img, conv, prediction, model)
            show(path_file_dir, label, probability, image_heatmap)
        else:
            show(path_file_dir, label, probability)


def predict(path_file: str, model, prepare_data, id_class: Dict,
            conv_output: bool = False) -> Tuple[int, str, torch.Tensor, Union[torch.Tensor, None], float]:
    """
    :param path_file: path to a file or a folder that contains images.
    :param model: torch Model.
    :param prepare_data: class inherited from torch.utils.data.Dataset.
    :param id_class: dict contains names and ID's.
    :param conv_output: get output from last conv layer ot not.
    :return: ID, class name, normalized image in the  form of a tensor (B, C, W, H), output last conv layer
            (B, C, W, H), probability of belonging to a class.
    """

    img_tensor = prepare_data.prepare_image(path_file).unsqueeze(0)

    predict_softmax = model(img_tensor)[1][0]
    predict_class = torch.argmax(predict_softmax).item()
    probability = predict_softmax[predict_class].item()

    lable = (list(id_class.keys())[list(id_class.values()).index(predict_class)])

    if conv_output:
        feutere_extractor, conv = get_information_layer(model=model, type_last_layer='conv', image=img_tensor)
        return predict_class, lable, img_tensor, conv, probability
    else:
        return predict_class, lable, img_tensor, None, probability


if __name__ == '__main__':
    # show_batch()

    # show_image_prediction('data/images/polyps', '/Users/kitenko/Downloads/model_best.pth', 'data/class_index.json',
    #                       apply_score_cam=True)
    # predict_test('data/images/cecum', '/Users/kitenko/Downloads/model_best.pth', 'data/class_index.json')

    count_metrics(weights='save_models/efficientnetv2_s_efficientnetv2_s_2022-03-01_18-33-03_shape-224-224/model_best.pth')

