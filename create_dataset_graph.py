import os
import matplotlib.pyplot as plt


# ==========================================
# DATASET PATHS
# ==========================================

training_dir = "dataset/brain_mri/Training"
testing_dir = "dataset/brain_mri/Testing"


# ==========================================
# CLASS NAMES
# ==========================================

classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ==========================================
# COUNT IMAGES
# ==========================================

training_counts = []
testing_counts = []


for class_name in classes:

    folder_name = class_name.lower().replace(" ", "")

    training_path = os.path.join(
        training_dir,
        folder_name
    )

    testing_path = os.path.join(
        testing_dir,
        folder_name
    )

    training_count = len(
        os.listdir(training_path)
    )

    testing_count = len(
        os.listdir(testing_path)
    )

    training_counts.append(
        training_count
    )

    testing_counts.append(
        testing_count
    )


# ==========================================
# PRINT DATASET INFORMATION
# ==========================================

print("===================================")
print("NEUROSCANAI DATASET DISTRIBUTION")
print("===================================")

for i in range(len(classes)):

    print(
        f"{classes[i]} : "
        f"Training = {training_counts[i]}, "
        f"Testing = {testing_counts[i]}"
    )


# ==========================================
# CREATE GRAPH
# ==========================================

x = range(len(classes))

width = 0.35


plt.figure(
    figsize=(10, 6)
)


plt.bar(
    [i - width / 2 for i in x],
    training_counts,
    width,
    label="Training"
)


plt.bar(
    [i + width / 2 for i in x],
    testing_counts,
    width,
    label="Testing"
)


# ==========================================
# GRAPH LABELS
# ==========================================

plt.title(
    "NeuroScanAI - Dataset Distribution"
)

plt.xlabel(
    "Brain MRI Classes"
)

plt.ylabel(
    "Number of Images"
)


plt.xticks(
    list(x),
    classes
)


plt.legend()

plt.grid(
    axis="y"
)


plt.tight_layout()


# ==========================================
# CREATE STATIC FOLDER
# ==========================================

os.makedirs(
    "static",
    exist_ok=True
)


# ==========================================
# SAVE GRAPH
# ==========================================

output_path = (
    "static/dataset_distribution.png"
)


plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nDataset distribution graph "
    "saved successfully!"
)

print(
    "Output:",
    output_path
)