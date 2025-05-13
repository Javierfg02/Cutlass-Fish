import re
import matplotlib.pyplot as plt

def parse_train_log(file_path):
    """
    Parses the training log to extract epoch numbers and corresponding training loss values.

    Args:
        file_path (str): Path to the training log file.

    Returns:
        tuple: A tuple containing two lists - epochs and losses.
    """
    epochs = []
    losses = []

    # Regex pattern to extract epoch number and loss
    epoch_pattern = re.compile(r'EPOCH (\d+)')
    loss_pattern = re.compile(r'total training loss ([\d\.]+)')

    file_path = 'Models/Base/train.log'

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Extract epoch
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                epochs.append(epoch)
            # Extract loss
            loss_match = loss_pattern.search(line)
            if loss_match:
                loss = float(loss_match.group(1))
                losses.append(loss)

    return epochs, losses

def plot_epochs_vs_loss(epochs, losses):
    """
    Plots epochs vs training loss.

    Args:
        epochs (list): List of epoch numbers.
        losses (list): List of loss values corresponding to each epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs[:-1], losses, linestyle='-', color='b')
    plt.xlabel("Epochs")
    plt.ylabel("Total Training Loss")
    plt.title("Epochs vs Total Training Loss")
    plt.grid(True)
    # plt.xticks(range(0, max(epochs)+1, max(epochs)//10))
    plt.show()

# Example usage
if __name__ == "__main__":
    log_file_path = 'train.log'  # Replace with the path to your train.log file
    epochs, losses = parse_train_log(log_file_path)
    plot_epochs_vs_loss(epochs, losses)
