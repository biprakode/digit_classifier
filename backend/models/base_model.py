from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input array of shape (n_samples, n_features)
            
        Returns:
            Array of predictions of shape (n_samples,)
        """
        raise NotImplementedError("Subclasses must implement predict()")