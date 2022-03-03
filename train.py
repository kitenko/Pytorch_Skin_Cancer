import os
import time
import json
from typing import Tuple
from datetime import datetime

import timm
import torch
from tqdm import tqdm
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import Model,  MetricsTorch, LoggingModel, CustomDataset, log_tensorboard, load_check_point, save_checkpoint
from config import (JSON_NAME, EPOCHS, LEARNING_RATE, SAVE_MODELS, INPUT_SHAPE_IMAGE, MODEL_NAME, WEIGHTS,
                    BACKBONE, LOGS, BATCH_SIZE, PATH_DATA, CLASS_JSON, SAVE_MODEL_EVERY_EPOCH, SAVE_BEST_MODEL,
                    SAVE_STATE_CURRENT_MODEL, NUMBER_CLASSES, PATH_LAST_STATE_MODEL, PATIENCE, GPU_NUM, NUM_WORKERS)


def train(data_path: str = PATH_DATA, input_shape_image: Tuple[int, int, int] = INPUT_SHAPE_IMAGE,
          last_loss: int = 5000, patience: int = PATIENCE, class_json: str = CLASS_JSON,
          save_model_every_epoch: bool = SAVE_MODEL_EVERY_EPOCH, model_name: str = MODEL_NAME, weights=WEIGHTS,
          gpu_num: int = GPU_NUM):
    """
    :param data_path: a path to the folder where the data is stored.
    :param input_shape_image:  input tensor, not considering batch size.
    :param last_loss: this variable is used for early stopping trainig.
    :param patience: how many times can the metric not be increased after the best value.
    :param class_json: name of json file, which containes names of classes and id.
    :param save_model_every_epoch: save the model every epoch or not.
    :param model_name: name of the model to be loaded.
    :param weights: use pre-trained weights or not.
    :param gpu_num: number of the hardware gpu that will be used for traing.
    :return: model object, current epoch, save path.
    """

    trigger_times = 0

    # load json with information index and class.
    with open(os.path.join(data_path, class_json), 'r') as file:
        cl_json = json.load(file)

    """-------------------------------------prepare folders for training-----------------------------------"""

    date_time_for_save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join('save_models', '{}_{}_{}_shape-{}'.format(MODEL_NAME, BACKBONE, date_time_for_save,
                                                                          str(input_shape_image[1]) + '-' +
                                                                          str(input_shape_image[2])))

    save_current_logs = os.path.join(save_path, LOGS)
    save_current_model = os.path.join(save_path, SAVE_MODELS)
    save_best_model = os.path.join(save_current_model, SAVE_BEST_MODEL)
    save_current_state_model = os.path.join(save_current_model, SAVE_STATE_CURRENT_MODEL)

    # create dirs
    for p in [save_path, save_current_model, save_current_logs, save_best_model, save_current_state_model]:
        os.makedirs(p, exist_ok=True)

    """---------------------------------------------------------------------------------------------------"""
    """-------------------------------------prepare data--------------------------------------------------"""

    train_data = CustomDataset(data_path=data_path, json_name=JSON_NAME, is_train='train',
                               image_shape=input_shape_image)
    val_data = CustomDataset(data_path=data_path, json_name=JSON_NAME, is_train='val',
                             image_shape=input_shape_image)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True,
                              persistent_workers=True)
    test_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True,
                             persistent_workers=True)

    training_samples_num = len(train_loader.dataset)
    testing_samples_num = len(test_loader.dataset)

    """---------------------------------------------------------------------------------------------------"""
    # create log instance
    log = LoggingModel(save_path_log=save_current_logs, list_augmentations=str(train_data.aug.transforms.transforms))

    device = torch.device('cuda:{}'.format(gpu_num) if torch.cuda.is_available() else 'cpu')

    # Load model
    loaded_model = timm.create_model(model_name, pretrained=weights, num_classes=NUMBER_CLASSES, in_chans=3)
    loaded_model = Model(loaded_model)
    summary(loaded_model, input_shape_image, device='cpu')

    # Initialization tensorboard
    tensorboard = SummaryWriter(log_dir=save_current_logs)
    tensorboard.add_graph(loaded_model, next(iter(train_loader))[0])
    loaded_model = loaded_model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(loaded_model.parameters(), lr=LEARNING_RATE)

    if PATH_LAST_STATE_MODEL is not None:
        loaded_model, optimizer, _ = load_check_point(PATH_LAST_STATE_MODEL, loaded_model, optimizer, device)

    start_time = time.time()

    best_val_f1 = 0.0
    best_epoch_num = 0

    metric = MetricsTorch(average='macro', device=device, accuracy=True, precision=True, recall=True, f1=True,
                          each_class=True)

    try:
        for epoch in range(1, EPOCHS + 1):
            # Set model to training mode.
            loaded_model.train()
            running_loss = 0.0

            print()

            with tqdm(train_loader, unit="batch") as tepoch:
                # Iterating over training data.
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Train_Epoch {epoch}")

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients.
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        # Forward.
                        outputs, soft_out = loaded_model(inputs)
                        _, preds = torch.max(soft_out, 1)
                        loss = loss_func(outputs, labels)

                        metric.training_step(labels, preds)
                        metric.training_step_each_class(labels, preds)

                        # Backward.
                        loss.backward()
                        optimizer.step()

                    # Get loss
                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / training_samples_num
                epoch_metrics = metric.training_step_compute()
                epoch_metrics_each_class = metric.training_step_each_class_compute()

            # Set model to evaluating mode.
            loaded_model.eval()
            running_loss= 0.0

            with tqdm(test_loader, unit="batch") as tepoch:

                # Iterating over testing data.
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Val_Epoch {epoch}")

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # Zero the parameter gradients.
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(False):
                        # Forward.
                        outputs, soft_out = loaded_model(inputs)
                        _, preds = torch.max(soft_out, 1)
                        loss = loss_func(outputs, labels)

                        metric.validation_step(labels, preds)
                        metric.validation_step_each_class(labels, preds)

                    # Get loss and corrects.
                    running_loss += loss.item() * inputs.size(0)

                val_epoch_loss = running_loss / testing_samples_num
                val_epoch_metrics = metric.validation_step_compute()
                val_epoch_metrics_each_class = metric.validation_step_each_class_compute()

                metric_print = log_tensorboard(tensorboard, train_metric=epoch_metrics, val_metric=val_epoch_metrics,
                                epoch_loss=epoch_loss, val_epoch_loss=val_epoch_loss,
                                train_metric_each_class=epoch_metrics_each_class,
                                val_metric_each_class=val_epoch_metrics_each_class, class_json=cl_json,
                                num_epoch=epoch, log=log, log_echa_metrics=True)

                print(metric_print)

            if float(metric_print['val_F1']) > best_val_f1:
                # save_ckp(loaded_model, True, save_current_state_model, save_best_model)
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': loaded_model.state_dict(),
                    'best_val_F1': float(metric_print['val_F1']),
                    'optimizer': optimizer.state_dict()
                }, True, save_current_state_model, save_best_model)
                best_val_f1 = float(metric_print['val_F1'])
                best_epoch_num = epoch

            else:
                # save_ckp(loaded_model, False, save_current_state_model, save_best_model)
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': loaded_model.state_dict(),
                    'best_val_F1': float(metric_print['val_F1']),
                    'optimizer': optimizer.state_dict()
                }, False, save_current_state_model, save_best_model)

            # Early stopping
            the_current_loss = val_epoch_loss
            if the_current_loss > last_loss:
                trigger_times += 1

                if trigger_times >= patience:
                    print('The current loss:', the_current_loss)
                    print('Early stopping!\nStart to test process.')
                    return loaded_model, epoch, save_current_model
            else:
                trigger_times = 0

            last_loss = the_current_loss

            metric.training_step_reset()
            metric.validation_step_reset()
            metric.training_step_each_class_reset()
            metric.validation_step_each_class_reset()

            if save_model_every_epoch:
                torch.save(loaded_model.state_dict(), os.path.join(save_current_model, '{:03d}_epoch.pth'.format(epoch)))

    except KeyboardInterrupt:
        print('\nStopped training.')
    finally:
        finish_time = time.time() - start_time
        tensorboard.close()
        print('\nTraining complete in {:.0f}m {:.0f}s.'.format(finish_time // 60, finish_time % 60))
        print('Best val F1: {:4f} (epoch {}).'.format(best_val_f1, best_epoch_num))


if __name__ == '__main__':
    model, epoch, save_current_model = train()
    torch.save(model.state_dict(), os.path.join(save_current_model, '{:03d}_epoch.pth'.format(epoch)))
