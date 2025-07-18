import sys
import os
import argparse

from models.resnet import ResNetMemeClassifier
from models.efficientnet import EfficientNetMemeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class_names = ['not cu', 'CU']


def get_model(name: str):
    if name == "resnet":
        return ResNetMemeClassifier()
    elif name == "efficientnet":
        return EfficientNetMemeClassifier()
    else:
        raise ValueError(
            f"Unknown model name: {name}. Choose 'resnet' or 'efficientnet'."
        )


def main():
    parser = argparse.ArgumentParser(description="Predict engagement from image")
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet", "efficientnet"],
        default="resnet",
        help="Model to use: 'resnet' or 'efficientnet'"
    )
    parser.add_argument("image", type=str, help="Path to input image")

    args = parser.parse_args()

    model = get_model(args.model)
    result = model.predict(args.image)
    print(f'Predicted class: {class_names[result[0]]}. Probs: {result[1]}')


if __name__ == '__main__':
    main()
