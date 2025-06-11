import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import constants
import seaborn as sns

from scipy.interpolate import interp1d

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def compare_errors():
    """
    Gather all of the error information from the current run to create a
    ridgeline plot with vertical orientation and vertical offsets.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    base_dir = f"./metrics/run-{constants.identifier}"

    # Keep track of the offsets for the ridgeline plot
    data_list = []
    labels = []
    # Collect all the data to plot
    for dir in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, dir)):
            for f in sorted(os.listdir(os.path.join(base_dir, dir))):
                if f.startswith("errors"):
                    loss, activation = tuple([s.strip() for s in dir.split("+")])
                    data = np.loadtxt(os.path.join(base_dir, dir, f))
                    labels.append(f"{loss} + {activation}")
                    data_list.append(data)

    # Set the offset and spacing between each plot
    spacing = 9  # Adjust the spacing between each plot
    initial_offset = spacing * (len(data_list) - 1)
    offset = initial_offset

    # Plot each dataset
    for data, label in zip(data_list, labels):
        # Use seaborn's kdeplot to create the ridgeline effect
        sns.kdeplot(data, ax=ax, shade=True, alpha=1.0,
                    label=label, vertical=False,
                    lw=2, fill=True, color=plt.cm.viridis(offset / initial_offset),
                    bw_adjust=0.5)  # Optional: adjust bandwidth for smoother plots

        # Apply the vertical offset by modifying the y-coordinates of the path vertices
        for collection in ax.collections[-1:]:
            for path in collection.get_paths():
                # Get the path's vertices
                vertices = path.vertices
                # Apply the vertical offset (adding to the y-values)
                path.vertices = vertices + np.array([0, offset])

        # Adjust the offset for each plot to stack them vertically
        offset -= spacing

    ax.set_ylim([0, 100])
    ax.set_ylabel('Frequency')  # y-axis represents error values
    ax.set_xlabel('Test Prediction Error $\\hat y - y$')
    ax.set_title('Frequency of Error')

    # Adding a legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.6)

    # Save the plot
    fig.savefig(os.path.join(base_dir, "errors-unified.png"), transparent=True)

def compare_losses():
    """
    Gather all of the training loss information from the current run to create
    a comparative plot of loss versus epoch for each combination.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    base_dir = f"./metrics/run-{constants.identifier}"
    n = len(list(filter(lambda dir: os.path.isdir(os.path.join(base_dir, dir)), os.listdir(base_dir))))
    i = n
    linestyle = ["--", "-.", "-"]
    for dir in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, dir)):
            loss_files = filter(lambda f: f.startswith("loss"), sorted(os.listdir(os.path.join(base_dir, dir))))
            for f in loss_files:
                loss, activation = tuple([s.strip() for s in dir.split("+")])
                data = np.loadtxt(os.path.join(base_dir, dir, f))
                interp_func = interp1d(data[:, 0], data[:, 1], kind='cubic')

                # Create a range of x values for smooth interpolation
                x_smooth = np.linspace(data[:, 0].min(), data[:, 0].max(), 50)
                y_smooth = interp_func(x_smooth)

                # Plot the smoothed data
                color_val = i / n
                ax.plot(x_smooth, y_smooth, label=f"{loss} + {activation}", color=plt.cm.viridis(color_val), linestyle=linestyle[i % 3])
                print(i, color_val)
                i -= 1
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), borderaxespad=0.)
    ax.grid(True)
    ax.set_ylim([0, 0.10])
    ax.set_ylabel('Loss')  # y-axis represents error values
    ax.set_xlabel('Epoch')
    ax.set_title('Loss vs Epoch')
    plt.subplots_adjust(bottom=0.5)
    fig.savefig(os.path.join(base_dir, "losses-unified.png"), transparent=True)
    plt.show()

def main():
    args = sys.argv
    constants.identifier = int(sys.argv[-1])
    compare_errors()
    compare_losses()

if __name__ == "__main__":
    main()
