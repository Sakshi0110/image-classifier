from .loader import load_test_data, load_train_data
from .model import model


if __name__ == "__main__":
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=load_train_data(), epochs=10, batch_size=20)

