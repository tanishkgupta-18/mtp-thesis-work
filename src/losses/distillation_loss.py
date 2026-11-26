
import tensorflow as tf


class KnowledgeDistillationLoss:
    """
    Knowledge Distillation loss for KD-SCA.

    L_total = (1-alpha) * L_CE
              + alpha * T^2 * L_KD

    Uses sparse integer labels and raw logits.
    """

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.7
    ):
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1.")

        self.temperature = temperature
        self.alpha = alpha

    def compute_kd_loss(
        self,
        y_true,
        student_logits,
        teacher_logits
    ):
        # Hard-label cross-entropy
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true,
            student_logits,
            from_logits=True
        )

        ce_loss = tf.reduce_mean(ce_loss)

        # Temperature-scaled log probabilities
        T = self.temperature

        teacher_log_probs = tf.nn.log_softmax(
            teacher_logits / T,
            axis=-1
        )

        student_log_probs = tf.nn.log_softmax(
            student_logits / T,
            axis=-1
        )

        teacher_probs = tf.exp(teacher_log_probs)

        # KL(P_teacher || P_student)
        kd_loss = tf.reduce_sum(
            teacher_probs *
            (teacher_log_probs - student_log_probs),
            axis=-1
        )

        kd_loss = tf.reduce_mean(kd_loss)

        # Dual supervision
        total_loss = (
            (1.0 - self.alpha) * ce_loss
            + self.alpha * (T ** 2) * kd_loss
        )

        return total_loss, ce_loss, kd_loss
