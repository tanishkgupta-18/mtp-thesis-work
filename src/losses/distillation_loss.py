import tensorflow as tf
from tensorflow.keras import losses


class KnowledgeDistillationLoss(losses.Loss):
    """Knowledge Distillation loss combining hard label cross-entropy and soft teacher supervision.

    L_total = (1 - alpha) * L_CE(y_true, y_student_pred) + alpha * (T^2) * L_KL(p_teacher_T, p_student_T)
    """

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.7,
        name: str = "kd_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss_fn = losses.CategoricalCrossentropy(from_logits=False)
        self.kl_div_fn = losses.KLDivergence()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute standard loss if called directly.

        For distillation training, use compute_kd_loss() with student and teacher logits/probabilities.
        """
        return self.ce_loss_fn(y_true, y_pred)

    def compute_kd_loss(
        self,
        y_true: tf.Tensor,
        student_logits: tf.Tensor,
        teacher_logits: tf.Tensor,
    ) -> tf.Tensor:
        """Compute dual-supervision KD loss.

        Args:
            y_true: One-hot encoded ground truth labels.
            student_logits: Unscaled output logits from student model.
            teacher_logits: Unscaled output logits from teacher model.

        Returns:
            tf.Tensor: Total combined distillation loss.
        """
        # Hard label cross-entropy loss on student predictions (T = 1)
        student_probs = tf.nn.softmax(student_logits, axis=-1)
        ce_loss = self.ce_loss_fn(y_true, student_probs)

        # Soft label distillation loss with temperature scaling
        soft_student = tf.nn.softmax(student_logits / self.temperature, axis=-1)
        soft_teacher = tf.nn.softmax(teacher_logits / self.temperature, axis=-1)
        kd_loss = self.kl_div_fn(soft_teacher, soft_student)

        # Total combined loss
        total_loss = (1.0 - self.alpha) * ce_loss + (self.alpha * (self.temperature ** 2)) * kd_loss
        return total_loss
