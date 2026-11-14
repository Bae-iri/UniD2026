"""Fixed linear diagnostic of sensor -> response predictability, without label training.

This supplements, and never changes, the primary neural comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from hierarchy_experiment import GAS, Preprocessor, split_frames, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    metadata = json.loads((args.run / "preprocessing.json").read_text())
    cache = json.loads((args.run / "image_cache_manifest.json").read_text())
    frames = split_frames(Path(config["data_root"]), config["split_seed"], config["dev_fraction"], config["smoke"])
    prep = Preprocessor(frames["fit"])
    manifest = pd.read_csv(args.run / "split_manifest.csv")
    for name, frame in frames.items():
        if manifest.loc[manifest.split == name, "sample_id"].tolist() != frame.sample_id.tolist():
            raise ValueError("Run manifest differs from reconstructed data")
    arrays = {}
    for name in ("fit", "validation"):
        q, gas, _ = prep.transform(frames[name])
        with np.load(Path(cache[name]).with_name("targets.npz")) as target_file:
            valid = target_file["valid"]
            image = (target_file["targets"] - np.asarray(metadata["image_mean"])) / np.asarray(metadata["image_std"])
        arrays[name] = (q.astype(np.float64), np.concatenate([gas[..., 0], image], axis=1),
                        np.concatenate([gas[..., 1] < 0.5, valid], axis=1))
    x, target, valid = arrays["fit"]
    xv, observed, observed_valid = arrays["validation"]
    prediction = np.zeros_like(observed)
    coefficients = np.zeros((target.shape[1], x.shape[1]))
    intercepts, means = np.zeros(target.shape[1]), np.zeros(target.shape[1])
    for column in range(target.shape[1]):
        usable = valid[:, column]
        if not usable.any():
            raise ValueError(f"No fit observations for target {column}")
        # Alpha is fixed in advance; no label, dev, or validation-based tuning.
        model = Ridge(alpha=1.0).fit(x[usable], target[usable, column])
        prediction[:, column] = model.predict(xv)
        coefficients[column], intercepts[column] = model.coef_, model.intercept_
        means[column] = target[usable, column].mean()
    sse = np.where(observed_valid, (prediction - observed) ** 2, 0)
    baseline = np.where(observed_valid, (means - observed) ** 2, 0)
    experiments = frames["validation"].experiment_id.to_numpy()
    groups = sorted(set(experiments))
    weights = np.random.default_rng(config["split_seed"] + 971).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=2000)
    metrics = {"method": "Ridge(alpha=1.0), fit only, no labels, no hyperparameter search",
               "role": "supplementary diagnostic; does not change the primary neural experiment"}
    for name, indices in (("gas", slice(0, len(GAS))), ("image", slice(len(GAS), None))):
        error, constant = sse[:, indices].sum(), baseline[:, indices].sum()
        count = observed_valid[:, indices].sum()
        group_error = np.asarray([sse[experiments == group, indices].sum() for group in groups])
        group_baseline = np.asarray([baseline[experiments == group, indices].sum() for group in groups])
        boot_error, boot_baseline = weights @ group_error, weights @ group_baseline
        boot_skill = 1 - boot_error / np.maximum(boot_baseline, 1e-12)
        low, high = np.quantile(boot_skill, [0.025, 0.975])
        metrics[name] = {"mse": float(error / count), "train_mean_mse": float(constant / count),
                         "skill_vs_mean": float(1 - error / constant),
                         "experiment_bootstrap_95_low": float(low), "experiment_bootstrap_95_high": float(high)}
    names = GAS + [f"image_{channel}_{row}_{col}" for channel in "RGB"
                   for row in range(config["image_grid"]) for col in range(config["image_grid"])]
    count = observed_valid.sum(0).clip(min=1)
    feature = pd.DataFrame({"feature": names, "mse": sse.sum(0) / count,
                            "train_mean_mse": baseline.sum(0) / count})
    feature["skill_vs_mean"] = 1 - feature.mse / feature.train_mean_mse.replace(0, np.nan)
    out = args.run / "sensor_only_probe"
    out.mkdir(exist_ok=True)
    write_json(out / "metrics.json", metrics)
    feature.to_csv(out / "features.csv", index=False)
    np.savez_compressed(out / "predictions.npz", prediction=prediction, target=observed, valid=observed_valid)
    np.savez(out / "ridge_model.npz", coefficients=coefficients, intercepts=intercepts)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
