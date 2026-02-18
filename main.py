from urban_flood_modeling import Settings, run_training


def main() -> None:
    settings = Settings()
    run_training(settings)


if __name__ == "__main__":
    main()
