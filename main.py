import shutil
from pathlib import Path

import loguru
import yaml
from ultralytics import YOLO

from src.utils import convert_images_to_jpg, create_yolo_dataset, xml2yolo


def preprocess(
    data_root_path: Path,
    val_split: float,
    test_split: float,
    force_preprocess: bool = False,
) -> None:
    """
    Convert data to YOLO ready format.
    """
    paths = {
        "imgs_path": data_root_path / "Image",
        "labels_path": data_root_path / "Markup",
        "yolo_labels_path": data_root_path / "labels",
        "dataset_path": data_root_path / "dataset",
    }

    if (
        not force_preprocess
        and paths["labels_path"].exists()
        and paths["dataset_path"].exists()
    ):
        loguru.logger.info("Dataset already exists, skipping preprocessing")
        return

    loguru.logger.info("Starting dataset preprocessing...")
    for path_name in ["yolo_labels_path", "dataset_path"]:
        shutil.rmtree(paths[path_name], ignore_errors=True)
        paths[path_name].mkdir()

    convert_images_to_jpg(paths["imgs_path"])

    name_to_label_mapping = xml2yolo(paths)

    create_yolo_dataset(paths, name_to_label_mapping, val_split, test_split)


def main():
    with open(Path("config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_root_path = Path(config["data_root_path"])
    imgsz = config["imgsz"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    val_split = config["val_part"]
    test_split = config["test_part"]
    force_preprocess = config["force_preprocess"]

    preprocess(data_root_path, val_split, test_split, force_preprocess=force_preprocess)
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(data_root_path / "dataset/dataset.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
    )


if __name__ == "__main__":
    main()
