import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_confusion_matrix(input_file, output_file):
    confusion_matrix = np.load(input_file)

    # a heatmap for each label's confusion matrix
    #since we are having multiple labels 94 stages) 
    for idx, matrix in enumerate(confusion_matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["False", "True"], yticklabels=["False", "True"])
        plt.title(f"Confusion Matrix for Label {idx}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        # each label's confusion matrix should be a seperate files 
        plt.savefig(output_file.replace(".png", f"_label_{idx}.png"))
        plt.close()

    print(f"Confusion matrices saved for all labels as PNG files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize confusion matrix for multi-label classification.")
    parser.add_argument("--input", required=True, help="Path to confusion matrix file (npy format).")
    parser.add_argument("--output", required=True, help="Path to save the confusion matrix PNG file.")

    args = parser.parse_args()
    visualize_confusion_matrix(args.input, args.output)
