import os
import re
import matplotlib.pyplot as plt


def parse_and_plot_logs(file_path):
    # Initialize lists to store our parsed metrics
    epochs = []
    train_losses = []
    train_accs = []
    val_accs = []

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Read and parse the log file line by line
    with open(file_path, "re") as f:
        for line in f:
            # Skip empty lines or header lines that don't contain '|'
            if "|" not in line:
                continue

            # Split line by the pipe character
            parts = [part.strip() for part in line.split("|")]

            if len(parts) >= 4:
                try:
                    # 1. Parse Epoch (Extracts the number before the slash)
                    epoch_match = re.match(r"(\d+)/", parts[0])
                    if not epoch_match:
                        continue
                    epoch = int(epoch_match.group(1))

                    # 2. Parse Loss
                    loss = float(parts[1])

                    # 3. Parse Train Accuracy (Strip '%' if present)
                    train_acc = float(parts[2].replace("%", ""))

                    # 4. Parse Validation Accuracy (Strip '%', '*', and spaces)
                    val_acc_clean = (
                        parts[3].replace("%", "").replace("*", "").strip()
                    )
                    val_acc = float(val_acc_clean)

                    # Append successfully parsed values
                    epochs.append(epoch)
                    train_losses.append(loss)
                    train_accs.append(train_acc)
                    val_accs.append(val_acc)

                except ValueError:
                    # Skip rows that fail to parse gracefully (e.g. headers)
                    continue

    if not epochs:
        print("No valid log lines were parsed. Check your file format.")
        return

    # --- PLOTTING CODE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ResNet18 Training Performance", fontsize=14, fontweight="bold")

    # Graph 1: Loss Curve
    ax1.plot(epochs, train_losses, color="#e74c3c", label="Train Loss", linewidth=2)
    ax1.set_title("Training Loss Curve")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Graph 2: Training vs Validation Accuracy
    ax2.plot(
        epochs, train_accs, color="#3498db", label="Train Accuracy", linewidth=2
    )
    ax2.plot(
        epochs,
        val_accs,
        color="#2ecc71",
        label="Val Accuracy",
        linewidth=2,
        linestyle="--",
    )
    ax2.set_title("Training vs. Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# --- HOW TO RUN IT ---
# Replace 'training_log.txt' with the actual path to your log file
log_file_path = r"C:\Users\shays\OneDrive\Desktop\Electrical Engineering first degree\EE 4th year\Final Project EE\resnet18_results\training_log.txt"
parse_and_plot_logs(log_file_path)