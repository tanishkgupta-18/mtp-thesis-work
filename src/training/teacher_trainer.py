
from typing import Dict, Any, Optional

import tensorflow as tf


class TeacherTrainer:
    """Trainer for the high-capacity profiling teacher CNN."""

    def __init__(
        self,
        model: tf.keras.Model,
        config: Dict[str, Any]
    ):
        self.model = model
        self.config = config

    def compile(self) -> None:
        """Compile teacher using integer labels and raw logits."""

        lr = self.config.get("learning_rate", 1e-4)
        optimizer_name = self.config.get("optimizer", "adam").lower()

        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr
            )
        elif optimizer_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=lr
            )
        else:
            optimizer = tf.keras.optimizers.get(optimizer_name)

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(
                    name="accuracy"
                )
            ],
        )

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        callbacks_list: Optional[list] = None,
    ) -> tf.keras.callbacks.History:
        """Train teacher with optional validation and callbacks."""

        batch_size = self.config.get("batch_size", 100)
        epochs = self.config.get("epochs", 50)

        validation_data = (
            (x_val, y_val)
            if x_val is not None and y_val is not None
            else None
        )

        return self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks_list,
            shuffle=True,
            verbose=1,
        )
