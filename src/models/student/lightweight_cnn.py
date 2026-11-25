import tensorflow as tf
from tensorflow.keras import layers, models


def build_student_cnn(
    input_shape: tuple = (700, 1),
    num_classes: int = 256,
    activation: str = "selu",
    return_logits: bool = True,
) -> tf.keras.Model:
    """Build the Lightweight Student CNN model from KD-SCA Paper (Table II).

    Layer Configuration:
        - Input Preprocessing: AvgPool1D(2, 2)
        - Conv Block 1: 32 filters, kernel 50, stride 1, padding 'same', BatchNorm, AvgPool(50, 50)
        - Conv Block 2: 16 filters, kernel 3, stride 1, padding 'valid', BatchNorm, AvgPool(2, 2)
        - Dense Layers: 2 x 10 neurons
        - Output Layer: 256 neurons
        - Total Parameters: 6,632 (for input shape 700, 1)

    Args:
        input_shape: Trace input dimensions, default (700, 1).
        num_classes: Number of classification targets, default 256.
        activation: Activation function ('selu' or 'relu').
        return_logits: If True, outputs unscaled logits (for distillation);
                       if False, outputs softmax probabilities.

    Returns:
        tf.keras.Model: Configured Keras Student CNN model.
    """
    initializer = "lecun_normal" if activation == "selu" else "he_uniform"

    inputs = layers.Input(shape=input_shape, name="trace_input")

    # Input Preprocessing
    x = layers.AveragePooling1D(pool_size=2, strides=2, name="input_avgpool")(inputs)

    # Conv Block 1
    x = layers.Conv1D(
        filters=32,
        kernel_size=50,
        strides=1,
        padding="same",
        activation=activation,
        kernel_initializer=initializer,
        name="conv1",
    )(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.AveragePooling1D(pool_size=50, strides=50, name="avgpool1")(x)

    # Conv Block 2
    x = layers.Conv1D(
        filters=16,
        kernel_size=3,
        strides=1,
        padding="valid",
        activation=activation,
        kernel_initializer=initializer,
        name="conv2",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.AveragePooling1D(pool_size=2, strides=2, name="avgpool2")(x)

    # Fully Connected Blocks
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(10, activation=activation, kernel_initializer=initializer, name="dense1")(x)
    x = layers.Dense(10, activation=activation, kernel_initializer=initializer, name="dense2")(x)

    # Output classification layer
    if return_logits:
        outputs = layers.Dense(num_classes, activation=None, name="logits")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="student_cnn")
    return model
