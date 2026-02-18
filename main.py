from urban_flood_modeling.infer import main as infer_main

from urban_flood_modeling.train import main as train_main


def main() -> None:
    train_main()
#    infer_main()


if __name__ == "__main__":
    main()
