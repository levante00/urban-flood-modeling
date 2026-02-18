import fire

from urban_flood_modeling.infer import main as infer_main
from urban_flood_modeling.train import main as train_main


def main(mode: str = "train", *overrides: str) -> None:
    mode_normalized = mode.strip().lower()
    compose_overrides = list(overrides)

    if mode_normalized == "train":
        train_main(compose_overrides)
        return
    if mode_normalized == "infer":
        infer_main(compose_overrides)
        return

    raise ValueError("mode must be either 'train' or 'infer'")


if __name__ == "__main__":
    fire.Fire(main)
