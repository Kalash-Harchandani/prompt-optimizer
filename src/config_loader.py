import json
from dataclasses import dataclass


@dataclass
class BudgetConfig:
    max_iterations: int
    max_cost_usd: float


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature_extraction: float
    temperature_optimizer: float


@dataclass
class Config:
    schema: str
    data_dir: str
    split_seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    budget: BudgetConfig
    llm: LLMConfig
    seed_prompt: str


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = json.load(f)

    budget = BudgetConfig(
        max_iterations=raw["budget"]["max_iterations"],
        max_cost_usd=raw["budget"]["max_cost_usd"]
    )

    llm = LLMConfig(
        provider=raw["llm"]["provider"],
        model=raw["llm"]["model"],
        temperature_extraction=raw["llm"]["temperature_extraction"],
        temperature_optimizer=raw["llm"]["temperature_optimizer"]
    )

    return Config(
        schema=raw["schema"],
        data_dir=raw["data_dir"],
        split_seed=raw["split_seed"],
        train_ratio=raw["train_ratio"],
        val_ratio=raw["val_ratio"],
        test_ratio=raw["test_ratio"],
        budget=budget,
        llm=llm,
        seed_prompt=raw["seed_prompt"]
    )