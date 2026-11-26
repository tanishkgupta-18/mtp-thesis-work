
from typing import Optional

import tensorflow as tf

from src.losses.distillation_loss import KnowledgeDistillationLoss


class KDScaDistiller(tf.keras.Model):
    """
    Knowledge Distillation Model for Side-Channel Analysis.

    The teacher remains frozen.
    Only student parameters are updated.
    """

    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        temperature: float = 3.0,
        alpha: float = 0.7,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.student = student
        self.teacher = teacher

        # Freeze teacher
        self.teacher.trainable = False

        self.temperature = temperature
        self.alpha = alpha

        self.kd_loss_fn = KnowledgeDistillationLoss(
            temperature=temperature,
            alpha=alpha
        )

        # Loss trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(
            name="total_loss"
        )

        self.ce_loss_tracker = tf.keras.metrics.Mean(
            name="ce_loss"
        )

        self.kd_loss_tracker = tf.keras.metrics.Mean(
            name="distill_loss"
        )

        # Classification accuracy
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.ce_loss_tracker,
            self.kd_loss_tracker,
            self.accuracy_tracker
        ]

    def compile(self, optimizer, **kwargs):

        super().compile(
            optimizer=optimizer,
            **kwargs
        )

    def train_step(self, data):

        x, y = data

        # Frozen teacher forward pass
        teacher_logits = self.teacher(
            x,
            training=False
        )

        teacher_logits = tf.stop_gradient(teacher_logits)

        with tf.GradientTape() as tape:

            # Student forward pass
            student_logits = self.student(
                x,
                training=True
            )

            # Compute combined KD loss
            total_loss, ce_loss, kd_loss = (
                self.kd_loss_fn.compute_kd_loss(
                    y,
                    student_logits,
                    teacher_logits
                )
            )

            # Include regularization losses if present
            if self.student.losses:
                total_loss += tf.add_n(self.student.losses)

        # Gradients only for student
        trainable_vars = self.student.trainable_variables

        gradients = tape.gradient(
            total_loss,
            trainable_vars
        )

        gradient_pairs = [
            (g, v)
            for g, v in zip(gradients, trainable_vars)
            if g is not None
        ]

        if not gradient_pairs:
            raise RuntimeError("No student gradients were calculated.")

        self.optimizer.apply_gradients(gradient_pairs)

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.ce_loss_tracker.update_state(ce_loss)
        self.kd_loss_tracker.update_state(kd_loss)

        self.accuracy_tracker.update_state(
            y,
            student_logits
        )

        return {
            "total_loss": self.total_loss_tracker.result(),
            "ce_loss": self.ce_loss_tracker.result(),
            "distill_loss": self.kd_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()
        }

    def test_step(self, data):

        x, y = data

        # Inference through frozen teacher and student
        teacher_logits = self.teacher(
            x,
            training=False
        )

        student_logits = self.student(
            x,
            training=False
        )

        total_loss, ce_loss, kd_loss = (
            self.kd_loss_fn.compute_kd_loss(
                y,
                student_logits,
                teacher_logits
            )
        )

        self.total_loss_tracker.update_state(total_loss)
        self.ce_loss_tracker.update_state(ce_loss)
        self.kd_loss_tracker.update_state(kd_loss)

        self.accuracy_tracker.update_state(
            y,
            student_logits
        )

        return {
            "total_loss": self.total_loss_tracker.result(),
            "ce_loss": self.ce_loss_tracker.result(),
            "distill_loss": self.kd_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result()
        }

    def call(self, inputs, training=False):

        return self.student(
            inputs,
            training=training
        )
