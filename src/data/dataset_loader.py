from typing import Tuple, Dict, Any, Optional
import h5py
import numpy as np


class ASCADDatasetLoader:
    """Dataset loader for ASCAD side-channel analysis HDF5 files."""

    def __init__(self, file_path: str, target_byte: int = 2):
        """Initialize dataset loader.

        Args:
            file_path: Path to the ASCAD .h5 file.
            target_byte: Index of the AES key/state byte targeted (default: 2).
        """
        self.file_path = file_path
        self.target_byte = target_byte

    def load_data(
        self,
        num_profiling: Optional[int] = None,
        num_attack: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Load profiling and attack traces along with labels and metadata from HDF5 file.

        Args:
            num_profiling: Maximum number of profiling traces to load (or None for all).
            num_attack: Maximum number of attack traces to load (or None for all).

        Returns:
            Dict containing:
                - 'x_profiling': Profiling traces array
                - 'y_profiling': Profiling labels array
                - 'x_attack': Attack traces array
                - 'y_attack': Attack labels array
                - 'metadata_profiling': Metadata for profiling traces
                - 'metadata_attack': Metadata for attack traces
        """
        with h5py.File(self.file_path, "r") as f:
            profiling_traces = f["Profiling_traces/traces"]
            profiling_labels = f["Profiling_traces/labels"]
            profiling_metadata = f["Profiling_traces/metadata"]

            attack_traces = f["Attack_traces/traces"]
            attack_labels = f["Attack_traces/labels"]
            attack_metadata = f["Attack_traces/metadata"]

            n_prof = num_profiling if num_profiling is not None else len(profiling_traces)
            n_att = num_attack if num_attack is not None else len(attack_traces)

            x_prof = np.array(profiling_traces[:n_prof], dtype=np.float32)
            y_prof = np.array(profiling_labels[:n_prof])
            meta_prof = np.array(profiling_metadata[:n_prof])

            x_att = np.array(attack_traces[:n_att], dtype=np.float32)
            y_att = np.array(attack_labels[:n_att])
            meta_att = np.array(attack_metadata[:n_att])

        return {
            "x_profiling": x_prof,
            "y_profiling": y_prof,
            "metadata_profiling": meta_prof,
            "x_attack": x_att,
            "y_attack": y_att,
            "metadata_attack": meta_att,
        }
