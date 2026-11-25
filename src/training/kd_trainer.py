from typing import Dict, Any, Optional
import tensorflow as tf
from src.losses.distillation_loss import KnowledgeDistillationLoss


class KDScaDistiller(tf.keras.Model):
    """Knowledge Distillation Model for Side-Channel Analysis (KD-SCA).

    Wraps teacher and student models, directing gradient updates exclusively to the student model.
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
        self.teacher.trainable = False  # Freeze teacher weights
        self.temperature = temperature
        self.alpha = alpha
        self.kd_loss_fn = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)

        # Trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.ce_loss_tracker = tf.keras.metrics.Mean(name="ce_loss")
        self.kd_loss_tracker = tf.keras.metrics.Mean(name="distill_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.ce_loss_tracker,
            self.kd_loss_tracker,
        ]

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data):
        x, y = data

        # Forward pass through frozen teacher (logits)
        teacher_logits = self.teacher(x, training=False)

        # Forward pass through student with gradient tape
        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)

            # Compute CE loss on true labels
            student_probs = tf.nn.softmax(student_logits, axis=-1)
            ce_loss = tf.keras.losses.categorical_crossentropy(y, student_probs)
            ce_loss = tf.reduce_mean(ce_loss)

            # Compute softened probability distributions for distillation
            soft_student = tf.nn.softmax(student_logits / self.temperature, axis=-1)
            soft_teacher = tf.nn.softmax(teacher_logits / self.temperature, axis=-1)
            kd_loss = tf.keras.losses.kl_divergence(soft_teacher, soft_student)
            kd_loss = tf.reduce_mean(kd_loss)

            # Dual supervision combined loss
            total_loss = (1.0 - self.alpha) * ce_loss + (self.alpha * (self.temperature ** 2)) * kd_loss

        # Apply gradients to student weights only
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.total_loss_tracker.update_state(total_loss)
        self.ce_loss_tracker.update_state(ce_loss)
        self.kd_loss_tracker.update_state(kd_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "ce_loss": self.ce_loss_tracker.result(),
            "distill_loss": self.kd_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data
        teacher_logits = self.teacher(x, training=False)
        student_logits = self.student(x, training=False)

        student_probs = tf.nn.softmax(student_logits, axis=-1)
        ce_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, student_probs))

        soft_student = tf.nn.softmax(student_logits / self.temperature, axis=-1)
        soft_teacher = tf.nn.softmax(teacher_logits / self.temperature, axis=-1)
        kd_loss = tf.reduce_mean(tf.keras.losses.kl_divergence(soft_teacher, soft_student))

        total_loss = (1.0 - self.alpha) * ce_loss + (self.alpha * (self.temperature ** 2)) * kd_loss

        self.total_loss_tracker.update_state(total_loss)
        self.ce_loss_tracker.update_state(ce_loss)
        self.kd_loss_tracker.update_state(kd_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "ce_loss": self.ce_loss_tracker.result(),
            "distill_loss": self.kd_loss_tracker.result(),
        }

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)
