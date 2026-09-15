import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]

training_loss = [
    0.8565,
    0.4740,
    0.3385,
    0.2637,
    0.2064
]

validation_loss = [
    0.5532,
    0.3984,
    0.3614,
    0.3100,
    0.2825
]

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    training_loss,
    label="Training Loss"
)

plt.plot(
    epochs,
    validation_loss,
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig(
    "loss_graph.png",
    dpi=300,
    bbox_inches="tight"
)

print("Loss graph created successfully!")