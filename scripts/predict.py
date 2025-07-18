import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from models.efficientnet import EfficientNetMemeRegressor
from models.resnet import ResNetMemeRegressor
from pathlib import Path

def get_model(name: str):
    if name == "resnet":
        return ResNetMemeRegressor()
    elif name == "efficientnet":
        return EfficientNetMemeRegressor()
    else:
        raise ValueError(f"Unknown model name: {name}. Choose 'resnet' or 'efficientnet'.")

def main():
    parser = argparse.ArgumentParser(description="Predict engagement from image")
    parser.add_argument("--model", type=str, choices=["resnet", "efficientnet"], default="resnet",
                        help="Model to use: 'resnet' or 'efficientnet'")
    parser.add_argument("image", type=str, help="Path to input image")

    args = parser.parse_args()

    model = get_model(args.model)

    likes, reposts, views = model.predict(args.image)

    print(f"👍 Likes:    {likes:.0f}")
    print(f"🔁 Reposts:  {reposts:.0f}")
    print(f"👀 Views:    {views:.0f}")


if __name__ == "__main__":
    main()
