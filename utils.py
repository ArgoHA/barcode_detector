import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import loguru
import numpy as np
import pandas as pd
import yaml
from skimage import io
from skmultilearn.model_selection import iterative_train_test_split


def convert_images_to_jpg(dir_path: Path) -> None:
    """
    Convert all images to .jpg format
    """
    all_files = [f.stem for f in dir_path.iterdir() if not f.name.startswith(".")]
    for filepath in dir_path.glob("*"):
        if filepath.suffix.lower() in [".tif", ".jpeg", ".png", ".tiff"]:
            image = io.imread(filepath, plugin="pil")
            io.imsave(filepath.with_suffix(".jpg"), image)
            filepath.unlink()

    jpg_files = [f.stem for f in dir_path.iterdir() if f.name.endswith(".jpg")]
    lost_files = set(all_files) - set(jpg_files)
    if not lost_files:
        loguru.logger.info(
            f"All files were converted to .jpg, total amount: {len(jpg_files)}"
        )
    else:
        loguru.logger.warning(
            f"Not converted to .jpg, amount: {len(lost_files)}, files: {lost_files}"
        )


def parse_xml(xml_file: Path) -> Dict:
    """
    Parse xml file.
    output: dict with lists of barcode types and points.
    """
    out = {"points": [], "barcode_type": []}
    with open(xml_file, "r") as f:
        xml_tree = ET.parse(f)

    root = xml_tree.getroot()

    for barcode in root.findall(".//Barcode"):
        barcode_type = barcode.get("Type")
        points = []

        for point in barcode.findall(".//Point"):
            x = point.get("X")
            y = point.get("Y")
            points.append((x, y))

        out["points"].append(points)
        out["barcode_type"].append(barcode_type)

    return out


def find_xyxy(coords: List[Tuple[str]]) -> List[int]:
    """
    Find left up point and right down point
    """
    x_coords, y_coords = zip(*coords)
    x_coords = list(map(int, x_coords))
    y_coords = list(map(int, y_coords))

    left_up_point = (min(x_coords), min(y_coords))
    right_down_point = (max(x_coords), max(y_coords))
    return [*left_up_point, *right_down_point]


def convert_coordinates(shape: Tuple, box: List) -> List[float]:
    """
    Convert coordinates of a bounding box to YOLO format
    """
    dw = shape[1]
    dh = shape[0]

    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]

    x = round(x / dw, 4)
    w = round(w / dw, 4)
    y = round(y / dh, 4)
    h = round(h / dh, 4)
    return [x, y, w, h]


def xml2yolo(paths: Dict[str, Path]) -> Dict[str, int]:
    """
    Convert default xml labels to YOLO format
    """
    all_files = [
        f.stem for f in paths["labels_path"].iterdir() if not f.name.startswith(".")
    ]
    name_to_label_mapping = {}
    for label_file in paths["labels_path"].iterdir():
        yolo_label_path = paths["yolo_labels_path"] / (label_file.stem + ".txt")
        img_file = paths["imgs_path"] / (label_file.stem + ".jpg")
        img = io.imread(img_file)

        raw_labels = parse_xml(label_file)

        for idx, _ in enumerate(raw_labels["points"]):
            label_name = raw_labels["barcode_type"][idx]

            if label_name not in name_to_label_mapping:
                name_to_label_mapping[label_name] = len(name_to_label_mapping)

            yolo_labels = [name_to_label_mapping[label_name]]
            points = raw_labels["points"][idx]

            bbox = find_xyxy(points)

            yolo_labels.extend(convert_coordinates((img.shape), bbox))

            with open(yolo_label_path, "a") as f:
                f.write(" ".join(map(str, yolo_labels)) + "\n")

    yolo_files = [
        f.stem
        for f in paths["yolo_labels_path"].iterdir()
        if not f.name.startswith(".")
    ]
    lost_files = set(all_files) - set(yolo_files)
    if not lost_files:
        loguru.logger.info(
            f"All labels were converted to YOLO format, total amount: {len(yolo_files)}"
        )
    else:
        loguru.logger.warning(
            f"Some labeles were not converted to YOLO format, total amount: {len(lost_files)}, files: {lost_files}"
        )

    return name_to_label_mapping


def move_data(
    data: Dict,
    paths: Dict[str, Path],
) -> None:
    """
    Copy images and labels to splitted folders
    input: data - dict with label names for each split part
    """
    parts_to_create = ["train", "val", "test"]
    split_paths = {}
    for part_name in parts_to_create:
        (paths["dataset_path"] / part_name).mkdir()

        split_paths[part_name] = paths["dataset_path"] / part_name

        (split_paths[part_name] / "images").mkdir()
        (split_paths[part_name] / "labels").mkdir()

    for part_name, part in data.items():
        for filename in part:
            filename = filename[0, 0].split(".jpg")[0]

            path_from = (paths["imgs_path"] / filename).with_suffix(".jpg")
            path_to = split_paths[part_name] / "images"
            shutil.copy(path_from, path_to)

            path_from = (paths["yolo_labels_path"] / filename).with_suffix(".txt")
            path_to = split_paths[part_name] / "labels"
            shutil.copy(path_from, path_to)


def split_dataset(
    paths: Dict[str, Path],
    val_part: float = 0.1,
    test_part: float = 0.1,
) -> None:
    """
    Multilabel split and folders preparation
    """
    all_images = [
        f.name for f in paths["imgs_path"].iterdir() if f.name.endswith(".jpg")
    ]
    labels = {}
    for filename in all_images:
        cur_labels = []
        try:
            with open((paths["yolo_labels_path"] / filename).with_suffix(".txt")) as f:
                lines = f.readlines()
                for line in lines:
                    cur_labels.append(line.split(" ")[0])
        except FileNotFoundError:
            continue
        labels[filename] = cur_labels

    # create a set of all unique values in the dictionary
    unique_values = set()
    for values in labels.values():
        unique_values.update(values)
    unique_values = sorted(list(unique_values))

    # create a new dictionary with the one-hot encoding for each key
    new_dict = {}
    for key, values in labels.items():
        one_hot_dict = {}
        for value in unique_values:
            if value in values:
                one_hot_dict[value] = 1
            else:
                one_hot_dict[value] = 0
        new_dict[key] = one_hot_dict

    # create a new DataFrame from the new dictionary
    df = pd.DataFrame.from_dict(new_dict, orient="index")
    df.reset_index(inplace=True)

    x = np.asmatrix(df.iloc[:, 0]).transpose()
    y = np.asmatrix(df.iloc[:, 1:].astype(dtype="float32"))

    data = {}

    data["train"], _, data["val"], y = iterative_train_test_split(
        x, y, test_size=val_part + test_part
    )
    data["val"], _, data["test"], _ = iterative_train_test_split(
        data["val"], y, test_size=1 / (val_part + test_part) * test_part
    )
    move_data(data, paths)

    loguru.logger.info(
        f'Tran cases: {len(data["train"])}, Val cases: {len(data["val"])}, Test cases: {len(data["test"])}'
    )


def create_yaml(name_to_label_mapping: Dict[str, int], dataset_path: Path) -> None:
    """
    Create a yaml file for YOLO dataset
    """
    yaml_data = {
        "train": str(dataset_path / "train"),
        "val": str(dataset_path / "val"),
        "test": str(dataset_path / "test"),
        "nc": len(name_to_label_mapping),
        "names": list(name_to_label_mapping.keys()),
    }

    with open(dataset_path / "dataset.yaml", "w") as f:
        yaml.dump(yaml_data, f)


def create_yolo_dataset(
    paths: Dict[str, Path],
    name_to_label_mapping: Dict,
    val_split: float,
    test_split: float,
) -> None:
    """
    Split dataset and create .yaml file for YOLO
    """
    shutil.rmtree(paths["dataset_path"], ignore_errors=True)
    paths["dataset_path"].mkdir()

    split_dataset(paths, val_split, test_split)

    create_yaml(name_to_label_mapping, paths["dataset_path"])
