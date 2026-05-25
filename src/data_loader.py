import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentSample:
    pdf_path: Path
    gold_json: dict
    filename: str


@dataclass
class DatasetSplits:
    train: list
    val: list
    test: list


def load_schema(schema_path: str) -> dict:
    with open(schema_path, "r") as f:
        return json.load(f)


def load_dataset(data_dir: str) -> list[DocumentSample]:
    data_path = Path(data_dir) / "pdf+gold"

    samples = []

    pdf_files = sorted(data_path.glob("*.pdf"))

    for pdf_path in pdf_files:
        base_name = pdf_path.stem

        gold_path = data_path / f"{base_name}.gold.json"

        if not gold_path.exists():
            print(f"Missing gold file for {base_name}")
            continue

        with open(gold_path, "r") as f:
            gold_json = json.load(f)

        sample = DocumentSample(
            pdf_path=pdf_path,
            gold_json=gold_json,
            filename=base_name
        )

        samples.append(sample)

    return samples


def create_splits(
    samples: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> DatasetSplits:

    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

    random.seed(seed)

    shuffled = samples.copy()
    random.shuffle(shuffled)

    total = len(shuffled)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return DatasetSplits(
        train=train,
        val=val,
        test=test
    )