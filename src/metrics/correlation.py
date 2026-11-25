import numpy as np
from scipy.stats import spearmanr
from typing import Tuple


def compute_model_correlation_matrix(
    student_probs: np.ndarray,
    teacher_probs: np.ndarray,
) -> np.ndarray:
    """Compute the Model Correlation (MC) matrix between student and teacher predictions.

    Args:
        student_probs: Softmax outputs of student model, shape (num_traces, num_classes).
        teacher_probs: Softmax outputs of teacher model, shape (num_traces, num_classes).

    Returns:
        np.ndarray: MC matrix of shape (num_traces, num_traces).
    """
    num_traces = student_probs.shape[0]
    mc_matrix = np.zeros((num_traces, num_traces), dtype=np.float32)

    for i in range(num_traces):
        for j in range(num_traces):
            corr, _ = spearmanr(student_probs[i, :], teacher_probs[j, :])
            mc_matrix[i, j] = corr

    return mc_matrix


def extract_diagonal_correlation(mc_matrix: np.ndarray) -> np.ndarray:
    """Extract diagonal elements of MC matrix representing same-trace alignment.

    Args:
        mc_matrix: Matrix of shape (N, N).

    Returns:
        np.ndarray: Diagonal elements array of length N.
    """
    return np.diag(mc_matrix)
