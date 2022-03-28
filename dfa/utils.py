from enum import Enum 

class ModelSettings(Enum):
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    BATCH = 200
    INPUT_DIMENSION = 784
    HIDDEN_DIMENSION = 800
    OUTPUT_DIMENSION = 10
    BINARISATION_THRESHOLD = 0.0
