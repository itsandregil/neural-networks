"""
Implementation of a general Loss Class.

Loss is a measure of the loss function, this value
means how much error the model has. So we ideally want it to be 0.

There are different loss function but they all work similarly for
different use-cases with Neural Networks

- First, we calculate the error for each sample
- Second, we calculate the mean value of the error
"""
import numpy as np


class Loss:
    def _forward(self, y_pred, y_true):
        """Forward pass for the loss function
        - y_pred (or y_hat): Model's predictions
        - y_true: Target labels
        """
        pass

    def calculate(self, y_pred, y_true):
        """
        Calculates the error for every sample and then returns
        the mean error.
        """
        # Calculate sample losses
        sample_losses = self._forward(y_pred, y_true)

        # Calculate mean error
        data_loss = np.mean(sample_losses)

        return data_loss


# Implementation of Cross-entropy Loss
class CategoricalCrossEntropy(Loss):
    def _forward(self, y_pred, y_true):
        """
        Forward pass for Categorical-Crossentropy
        """

        # Number of sample in a batch
        number_samples = len(y_true)

        # Clip data to prevent division by 0
        # Clip both side to not drag mean towards any value
        # From 1e-7 (almost 0) to almost 1 (1 - 1e-7)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - for sparse labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(number_samples), y_true]

        # Mask values for target values - for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods
