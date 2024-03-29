import argparse
import ast
from glob import glob
from itertools import permutations
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
import torch

from tqdm import tqdm

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(path, "samples.csv"),
        sep=",",
        index_col=0,
        converters={"Tokens": ast.literal_eval},
    )

    # Calculate sequence lengths
    df['SeqLength'] = df.Tokens.apply(lambda item: item.index('[SEP]') if '[SEP]' in item else item.index('</s>') if '</s>' in item else len(item))

    # Get indices of rows to drop
    indices_to_drop = df[df['SeqLength'] > 20].index
    # print(f"Indices to drop: {indices_to_drop}")

    # Drop row from df
    # print(get_longest_seq(df))
    df = df.drop(indices_to_drop)
    # print(get_longest_seq(df))

    # Load the tensor
    t = torch.load(os.path.join(path, "attentions.pt"))

    # Create a mask where each element is True if its index is not in indices_to_drop
    mask = torch.ones(len(t), dtype=torch.bool)
    mask[indices_to_drop] = False

    # Filter the tensor using the mask
    filtered_t = t[mask]

    # Add the extra dimension
    df["Attentions"] = filtered_t.unsqueeze(1)

    return df

def get_longest_seq(df: pd.DataFrame) -> int:
    df['SeqLength'] = df.Tokens.apply(lambda item: item.index('[SEP]') if '[SEP]' in item else item.index('</s>') if '</s>' in item else len(item))
    max_seq_length = df['SeqLength'].max()

    return max_seq_length

def plot_sim_per_layer(
    A: pd.DataFrame, B: pd.DataFrame, sim_fn: callable, desc: str, data_path: str
):
    sim_arr: np.ndarray = sim_fn(A, B).squeeze()

    n_layers, n_heads, q, v = sim_arr.shape

    longest_seq = get_longest_seq(A)
    if longest_seq == 0:
        longest_seq = q

    for layer in tqdm(
        range(n_layers),
        desc=f"plot_sim_per_layer: {sim_fn.__name__} {os.path.basename(data_path)}",
        leave=False,
    ):
        # setup grid for heatmaps of heads
        fig, axs = plt.subplots(n_layers // 3, 3, figsize=(30, 30), constrained_layout=True)
        axs = axs.flatten()

        for head in range(n_heads):
            ax = axs[head]

            # Adjust the slicing
            values = sim_arr[layer, head, :longest_seq, :longest_seq]

            # create heatmap
            heatmap = sns.heatmap(
                values,
                ax=ax,
                cmap=sns.cm.rocket,  # cmap=sns.cm.rocket_r, # reversed
            )
            heatmap.set_title(f"Layer {layer+1}, Head {head+1}")
            heatmap.set_xlabel("Tokens")
            heatmap.set_ylabel("Tokens")

        # write to file
        filepath = os.path.join(
            data_path, sim_fn.__name__, f"layer_{str(layer+1).rjust(2, '0')}.png"
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        plt.savefig(filepath)
        plt.close()

def plot_sim_per_head(
    A: pd.DataFrame, B: pd.DataFrame, sim_fn: callable, desc: str, data_path: str
):
    sim_arr: np.ndarray = sim_fn(A, B).squeeze()

    n_layers, n_heads, q, v = sim_arr.shape

    longest_seq = get_longest_seq(A)
    if longest_seq == 0:
        longest_seq = q

    for head in tqdm(
        range(n_heads),
        desc=f"plot_sim_per_head: {sim_fn.__name__} {os.path.basename(data_path)}",
        leave=False,
    ):
        # setup grid for heatmaps of layers
        fig, axs = plt.subplots(n_layers // 3, 3, figsize=(30, 30), constrained_layout=True)
        axs = axs.flatten()

        for layer in range(n_layers):
            ax = axs[layer]

            # Adjust the slicing
            values = sim_arr[layer, head, :longest_seq, :longest_seq]

            # create heatmap
            heatmap = sns.heatmap(
                values,
                ax=ax,
                cmap=sns.cm.rocket,  # cmap=sns.cm.rocket_r, # reversed
            )
            heatmap.set_title(f"Head {head+1}, Layer {layer+1}")
            heatmap.set_xlabel("Tokens")
            heatmap.set_ylabel("Tokens")

        # write to file
        filepath = os.path.join(
            data_path, sim_fn.__name__, f"head_{str(head+1).rjust(2, '0')}.png"
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        plt.savefig(filepath)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to the model directory as glob pattern.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="*",
        help="Path to the dataset directory as glob pattern.",
    )

    parser.add_argument(
        "--comparators",
        type=str,
        required=True,
        nargs="+",
        help="Import path of comparator function or class.",
    )

    args = parser.parse_args()

    try:
        comparators = []
        for comparator in args.comparators:
            comparator = comparator.split(".")
            comparator_module = __import__(
                ".".join(comparator[:-1]), fromlist=[comparator[-1]]
            )
            comparators.append(getattr(comparator_module, comparator[-1]))
        args.comparators = comparators
    except ImportError:
        raise ImportError(f"Could not import comparator {args.comparator}.")

    return args


def main():
    sns.set_theme(font_scale=2.5)

    args = parse_args()
    model_name: str = args.model_name
    dataset_directory: str = args.dataset_name
    comparators: list[callable] = args.comparators
    variants = []
    num_permutations = 0
    for cur, subdirs, files in os.walk(*glob(dataset_directory)):
        if "variant_00" not in subdirs:
            continue

        subdirs: list[str] = [dir for dir in subdirs if dir.startswith("variant_")]
        for variant in subdirs:
            variants.append(
                {
                    "path": cur,
                    "number": int(variant[-2:]),
                    "df": load(os.path.join(cur, variant)),
                }
            )

        # update number of permutations
        num_permutations += len(list(permutations(range(len(subdirs)), 2)))

    known_permutations = set()
    for left, right in tqdm(permutations(variants, 2), total=num_permutations):
        left_path, left_number, left_df = left.values()
        right_path, right_number, right_df = right.values()

        # only compare same sequence lengths
        if (
            os.path.basename(left_path) != os.path.basename(right_path)
            or (left_number, right_number) in known_permutations
            or (right_number, left_number) in known_permutations
        ):
            continue

        known_permutations.add(
            (
                left_number,
                right_number,
            )
        )

        for comparator in comparators:
            plot_sim_per_head(
                left_df,
                right_df,
                comparator,
                comparator.__doc__,
                os.path.join(
                    os.sep if model_name.startswith(os.sep) else "",
                    model_name,
                    left_path,
                    "png",
                    "per_head",
                    f"{left_number}{right_number}",
                ),
            )

            plot_sim_per_layer(
                left_df,
                right_df,
                comparator,
                comparator.__doc__,
                os.path.join(
                    os.sep if model_name.startswith(os.sep) else "",
                    model_name,
                    left_path,
                    "png",
                    "per_layer",
                    f"{left_number}{right_number}",
                ),
            )


if __name__ == "__main__":
    main()
