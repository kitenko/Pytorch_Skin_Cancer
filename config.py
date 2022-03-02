BATCH_SIZE = 64
NUMBER_CLASSES = 7
INPUT_SHAPE_IMAGE = (3, 224, 224)
EPOCHS = 500
AUGMENTATION_DATA = False
PATIENCE = 40
GPU_NUM = 0
NUM_WORKERS = 8

PATH_DATA = 'data'
JSON_NAME = 'index.json'
CLASS_JSON = 'index.json'
IMAGES_PATH = 'images'
MASKS_PATH = 'masks'

DATASETS = ['images']

LEARNING_RATE = 0.0001
BACKBONE = 'efficientnetv2_s'
WEIGHTS = True
OUTPUT_ACTIVATION = 'softmax'
LAYER_ACTIVATION = 'relu'
MODEL_NAME = 'efficientnetv2_s'
SAVE_MODEL_EVERY_EPOCH = False

LOGS = 'logs'
SAVE_MODELS = 'save_models'
SAVE_STATE_CURRENT_MODEL = 'save_state'
SAVE_BEST_MODEL = 'best_model'

PATH_LAST_STATE_MODEL = None