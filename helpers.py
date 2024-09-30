## helper functions and accessors to make lookups and assignments more consistent 
from typing import Union, List, Optional

## Here we have removed should_execute from the tracable model class entirely and placed in an external method
def should_execute(epoch: int, num_epochs: int, epoch_interval: Optional[Union[int, float, List[Union[int, float]]]] = None):
    """
    Determines whether recording logic should execute based on the current epoch and epoch_interval.
    Validates the input for epoch_interval and ensures recording logic is executed correctly.

    Args:
        epoch (int): The current epoch number.
        num_epochs (int): The total number of epochs.
        epoch_interval (Union[int, float, list, None]): The interval or list of epochs at which to execute the recording logic.

    Returns:
        bool: True if the recording logic should execute, False otherwise.

    Raises:
        ValueError: If epoch_interval is invalid (e.g., greater than num_epochs, negative but not -1).
    """

    # Validation: Check if epoch_interval is greater than num_epochs
    if isinstance(epoch_interval, (int, float)):
        if epoch_interval > num_epochs:
            raise ValueError(f"epoch_interval {epoch_interval} cannot be greater than the total number of epochs {num_epochs}.")
        if epoch_interval < -1:
            raise ValueError(f"Invalid epoch_interval {epoch_interval}. Must be -1, a positive integer, or a list of integers.")

    # Validation: If epoch_interval is a list, sort it and validate values
    if isinstance(epoch_interval, list):
        epoch_interval = sorted(epoch_interval)
        if any(e > num_epochs for e in epoch_interval):
            raise ValueError(f"Some values in epoch_interval {epoch_interval} are greater than the total number of epochs {num_epochs}.")
        if any(e < 0 for e in epoch_interval):
            raise ValueError(f"All values in epoch_interval list must be non-negative, but got {epoch_interval}.")

    # Logic for determining if the recording should execute
    if epoch_interval == -1:
        return epoch == num_epochs - 1  # Execute only at the last epoch
    elif isinstance(epoch_interval, list):
        return epoch in epoch_interval  # Execute only at specified epochs
    elif epoch_interval is not None:
        return epoch % epoch_interval == 0  # Execute at every nth epoch
    return True  # Execute at every epoch if epoch_interval is None

# Example usage in training loop:
# num_epochs = 20
# epoch_interval = [15, 5, 10]  # Example interval list (unsorted)

# for epoch in range(num_epochs):
#     if should_execute(epoch, num_epochs, epoch_interval):
#         # Insert the logic for recording weights, biases, or activations
#         print(f"Recording at epoch {epoch}")

#     # Training code...

#     if should_execute(epoch, num_epochs, epoch_interval):
#         print(f"Finished recording for epoch {epoch}")