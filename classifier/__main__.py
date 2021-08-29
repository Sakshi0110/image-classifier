from .loader import load_test_data, load_train_data, load_val_data
from .model import model


if __name__ == "__main__":
    model.fit(x=load_train_data(), epochs=10, batch_size=20, steps_per_epoch=20000,
              validation_data=load_val_data(), validation_steps=1000)
    model.save("weights.h5")
    results = model.predict(load_test_data(), steps=12500)
    with open("submissions.csv", "w") as f:
        f.write("imageID, ans\n")
        for idx, result in enumerate(results):
            f.write(f"{idx + 1},{'Cat' if result < 0.5 else 'Dog'}\n")

