# %%
import os
import torch
import pandas as pd
import numpy as np
import scipy
import glob
import json

from helpers import get_edit_positions, get_dataset, unmask_item
from transformers import AutoTokenizer

# %%
# https://gist.github.com/jlherren/d97839b1276b9bd7faa930f74711a4b6

MODEL_NAME = (
    os.getenv("MODEL_NAME")
    or "/media/data/models/roberta-base"  # "sjrhuschlee/flan-t5-base-squad2"  # "/media/data/models/flan-t5-base"
)  # "/media/data/models/bert-base-cased"
ATTENTION_TYPE = (
    os.getenv("ATTENTION_TYPE") or "attentions"  # "encoder_attentions"
)  # "/media/data/models/bert-base-cased"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, return_tensors="pt")


# %%
def max_seq_length(tokens):
    end_tokens = ["[SEP]", "</s>", "[PAD]", "<pad>"]
    return min(
        tokens.index(end_token) if end_token in tokens else len(tokens)
        for end_token in end_tokens
    )


def remove_padding(tokens):
    seq_length = max_seq_length(tokens)
    return tokens[:seq_length]


def get_target_attentions(item):
    try:
        positive: torch.Tensor = item.attentions_sentence.clone()[
            :, :, :, item.tokenized_edited_positions_in_positive
        ]
        negative: torch.Tensor = item.attentions_negative.clone()[
            :, :, :, item.tokenized_edited_positions_in_negative
        ]
        negative_with_original: torch.Tensor = item.attentions_negative.clone()[
            :, :, :, item.tokenized_edited_positions_in_negative_with_original
        ]
    except IndexError:
        print(item)
        print(item.tokenized_edited_positions_in_positive)
        print(item.tokenized_edited_positions_in_negative)
        print(item.attentions_sentence.shape)
        print(item.attentions_negative.shape)
        raise
    return positive, negative, negative_with_original


# %%
def load(dataset_path: str) -> pd.DataFrame:
    if "LAMA_negated" in dataset_path:
        for df in get_dataset():
            dataset = df.iloc[0,]["dataset"]
            for attention_type in glob.glob(
                f"/media/data/thielen/ba/negation_datasets/{os.path.basename(MODEL_NAME)}/{ATTENTION_TYPE}/LAMA_negated/{dataset}/*/*"
            ):
                if not os.path.exists(f"{attention_type}/variant_00/attentions.pt"):
                    # print(f"Skipping {attention_type}")
                    continue

                # print(f"Loading {attention_type}")
                df["attentions_sentence"] = [
                    row
                    for row in torch.load(
                        os.path.join(attention_type, "variant_00", "attentions.pt")
                    )
                ]

                df["attentions_negative"] = [
                    row
                    for row in torch.load(
                        os.path.join(attention_type, "variant_01", "attentions.pt")
                    )
                ]
                if "t5" not in MODEL_NAME:
                    df["pooler_positive"] = [
                        row
                        for row in torch.load(
                            os.path.join(
                                attention_type, "variant_00", "pooler_outputs.pt"
                            )
                        )
                    ]
                    df["pooler_negative"] = [
                        row
                        for row in torch.load(
                            os.path.join(
                                attention_type, "variant_01", "pooler_outputs.pt"
                            )
                        )
                    ]
                df["dataset_path"] = attention_type

                yield df
                # return
    else:
        # return
        df = pd.DataFrame()

        for attention_type in glob.glob(f"{dataset_path}/*/*/*"):
            if not os.path.exists(f"{attention_type}/variant_00/attentions.pt"):
                print(f"Skipping {attention_type}")
                continue

            # print(f"Loading {attention_type}")

            df["positive_sentences"] = pd.read_csv(
                f"{attention_type}/variant_00/samples.csv"
            )["Sentence"]
            df["negative_sentences"] = pd.read_csv(
                f"{attention_type}/variant_01/samples.csv"
            )["Sentence"]

            df["attentions_sentence"] = [
                row for row in torch.load(f"{attention_type}/variant_00/attentions.pt")
            ]
            df["attentions_negative"] = [
                row for row in torch.load(f"{attention_type}/variant_01/attentions.pt")
            ]

            if "t5" not in MODEL_NAME:
                df["pooler_positive"] = [
                    row
                    for row in torch.load(
                        os.path.join(attention_type, "variant_00", "pooler_outputs.pt")
                    )
                ]
                df["pooler_negative"] = [
                    row
                    for row in torch.load(
                        os.path.join(attention_type, "variant_01", "pooler_outputs.pt")
                    )
                ]
            df["dataset_path"] = attention_type

            print(f"Loaded {len(df)} samples from {attention_type}")

            yield df


# %%
dfs = []
for dataset_path in glob.glob(
    f"/media/data/thielen/ba/negation_datasets/{os.path.basename(MODEL_NAME)}/{ATTENTION_TYPE}/*"
):
    # print(f"Loading {dataset_path}")
    dfs.extend(load(dataset_path))  # extend implicitly iterates over the generator

for i in range(len(dfs)):
    if "masked_sentences" in dfs[i].columns:
        dfs[i]["positive_sentences"] = dfs[i].apply(
            lambda item: unmask_item(item, "masked_sentences"),
            axis=1,
        )
        dfs[i]["negative_sentences"] = dfs[i].apply(
            lambda item: unmask_item(item, "negated"),
            axis=1,
        )

print(f"Loaded {len(dfs)} dataframes")


# %%
def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            # results.append((ind,ind+sll-1))
            results.append(list(range(ind, ind + sll)))

    return [x for y in results for x in y]


def maybe_add_original_positions(item):
    roberta_prefix = "Ġ"
    t5_prefix = "▁"
    negators = ["not", "cannot"]
    negators = negators + [
        f"{prefix}{negator}"
        for negator in negators
        for prefix in [roberta_prefix, t5_prefix]
    ]
    ret = []
    for target_idx in item.tokenized_edited_positions_in_negative:
        ret.append(target_idx)
        tokens = remove_special_tokens(item.tokenized_negative_sentences)
        if target_idx >= len(tokens):
            continue
        if (
            remove_special_tokens(item.tokenized_negative_sentences)[target_idx]
            in negators
        ):
            ret.append(target_idx + 1)

    return list(sorted(set(ret)))


import torch.nn.functional as F


def calculate_cosine_similarity(row):
    tensor1 = row["attentions_sentence"].view(12 * 12, 32 * 32)
    tensor2 = row["attentions_negative"].view(12 * 12, 32 * 32)
    cosine_similarity = F.cosine_similarity(tensor1, tensor2)
    return cosine_similarity.mean().item()


def calculate_euclidean_distance(row):
    tensor1 = row["attentions_sentence"].view(12 * 12, 32 * 32)
    tensor2 = row["attentions_negative"].view(12 * 12, 32 * 32)
    euclidean_distance = F.pairwise_distance(tensor1, tensor2)
    return euclidean_distance.mean().item()


def tokenize_with_special(input):
    return TOKENIZER.tokenize(
        input,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=32,
    )


def remove_special_tokens(input):
    return list(
        filter(
            lambda x: x not in TOKENIZER.all_special_tokens,
            input,
        )
    )


def tokenize_without_special(input):
    return remove_special_tokens(tokenize_with_special(input))


for i in range(len(dfs)):
    dfs[i]["tokenized_positive_sentences"] = dfs[i].apply(
        lambda s: tokenize_with_special(s["positive_sentences"]), axis=1
    )
    dfs[i]["tokenized_negative_sentences"] = dfs[i].apply(
        lambda s: tokenize_with_special(s["negative_sentences"]), axis=1
    )

    dfs[i]["tokenized_edited_positions_in_negative"] = dfs[i].apply(
        lambda s: get_edit_positions(
            remove_special_tokens(s["tokenized_positive_sentences"]),
            remove_special_tokens(s["tokenized_negative_sentences"]),
        )[: s["attentions_sentence"].shape[-1]],
        axis=1,
    )
    dfs[i]["tokenized_edited_tokens_in_negative"] = dfs[i].apply(
        lambda s: [
            remove_special_tokens(s["tokenized_negative_sentences"])[idx]
            for idx in s["tokenized_edited_positions_in_negative"]
            if idx < len(remove_special_tokens(s["tokenized_negative_sentences"]))
        ],
        axis=1,
    )

    dfs[i]["tokenized_edited_positions_in_positive"] = dfs[i].apply(
        lambda s: get_edit_positions(
            remove_special_tokens(s["tokenized_negative_sentences"]),
            remove_special_tokens(s["tokenized_positive_sentences"]),
        )[: s["attentions_sentence"].shape[-1]],
        axis=1,
    )

    dfs[i]["tokenized_edited_positions_in_negative_with_original"] = dfs[i].apply(
        maybe_add_original_positions, axis=1
    )

    dfs[i] = dfs[i][
        dfs[i]["tokenized_edited_tokens_in_negative"].apply(len) > 0
    ].reset_index(drop=True)

    dfs[i]["cosine_sim_full_att"] = dfs[i].apply(calculate_cosine_similarity, axis=1)
    dfs[i]["euclidean_distance_full_att"] = dfs[i].apply(
        calculate_euclidean_distance, axis=1
    )
    dfs[i][["seq_len_pos", "seq_len_neg"]] = dfs[i][
        ["tokenized_positive_sentences", "tokenized_negative_sentences"]
    ].apply(lambda x: [max_seq_length(i) for i in x])

    dfs[i]["model_name"] = MODEL_NAME

# %%
## filter frames
legal_tokens = [
    "no",
    "No",
    "not",
    "does not",
    "do not",
    "did not",
    "don't",
    "doesn't",
    "are not",
    "isn't",
    "can't",
    "cannot",
    "is not",
]

legal_tokens = TOKENIZER.batch_decode(
    TOKENIZER.batch_encode_plus(legal_tokens)["input_ids"], skip_special_tokens=True
)
if "roberta" in MODEL_NAME:
    legal_tokens.extend([f"Ġ{token}" for token in legal_tokens])
elif "t5" in MODEL_NAME:
    legal_tokens.extend([f"▁{token}" for token in legal_tokens])  # not a regular '_'
legal_tokens = [[tok] for tok in legal_tokens]


def starts_with_legal_token(value):
    return any(value[: len(token)] == token for token in legal_tokens)


for i in range(len(dfs)):
    print("Shape before filtering:", dfs[i].shape)
    index = dfs[i][
        dfs[i].positive_sentences
        == "Three humans are in an airport. One is on the ground, another is next to the window, and the last one is vertical on his two feet with a bag next to him."
    ]
    if len(index):
        dfs[i] = dfs[i].drop(index.index)

    index = dfs[i][
        dfs[i].tokenized_edited_positions_in_positive.map(max)
        > dfs[i].attentions_sentence.map(lambda x: x.shape[-1])
    ]
    if len(index):
        dfs[i] = dfs[i].drop(index.index)

    filtered_df = dfs[i][
        dfs[i].tokenized_edited_tokens_in_negative.apply(starts_with_legal_token)
    ]
    filtered_df.reset_index(drop=True, inplace=True)
    print("Shape after filtering:", filtered_df.shape)
    dfs[i] = filtered_df


# %%
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        return obj


def get_values(func, *args, **kwargs):
    keys = [
        key
        for key in dir(func(*args, **kwargs))
        if not key.startswith("_") and key not in ["count", "index"]
    ]
    values = [getattr(func(*args, **kwargs), key) for key in keys]
    values = [value() if callable(value) else value for value in values]
    values = [make_serializable(value) for value in values]

    return {key: value for key, value in zip(keys, values)}


def serialize_value_counts(obj):
    return {str(key): int(value) for key, value in obj.value_counts().items()}


def get_attention_on_special_tokens(item):
    mask_sentence = np.in1d(
        item.tokenized_positive_sentences, TOKENIZER.all_special_tokens
    )
    mask_negative = np.in1d(
        item.tokenized_negative_sentences, TOKENIZER.all_special_tokens
    )
    res_1 = item.attentions_sentence.clone()
    res_2 = item.attentions_negative.clone()

    if any(mask_sentence):
        res_1[:, :, :, ~mask_sentence] = 0
    if any(mask_negative):
        res_2[:, :, :, ~mask_negative] = 0

    return (res_1, res_2)


def split_attentions_by_negative_positions(item):
    mask = pd.Series(
        i in item.tokenized_edited_positions_in_negative
        for i in range(item.attentions_sentence.shape[-1])
    )

    return [
        item.attentions_negative[:, :, :, mask],
        item.attentions_negative[:, :, :, ~mask],
    ]


def split_attentions_by_negative_with_original_positions(item):
    mask = pd.Series(
        i in item.tokenized_edited_positions_in_negative_with_original
        for i in range(item.attentions_sentence.shape[-1])
    )

    return [
        item.attentions_negative[:, :, :, mask],
        item.attentions_negative[:, :, :, ~mask],
    ]


def split_attentions_by_positive_positions(item):
    mask = pd.Series(
        i in item.tokenized_edited_positions_in_positive
        for i in range(item.attentions_sentence.shape[-1])
    )

    return [
        item.attentions_sentence[:, :, :, mask],
        item.attentions_sentence[:, :, :, ~mask],
    ]


outputs = []
target_dfs = []
standard_metrics = {
    "mean": "mean",
    "std": "std",
    "min": "min",
    "max": "max",
    "25%": lambda x: x.quantile(0.25),
    "median": "median",
    "75%": lambda x: x.quantile(0.75),
}

for df in dfs:
    print(f"Processing {df.iloc[0,]['dataset_path']}")

    target_df = df.apply(get_target_attentions, axis=1, result_type="expand")
    target_df.columns = [
        "att_positive_tok",
        "att_negative_tok",
        "att_negative_tok_with_original",
    ]
    target_df[
        [
            "att_sum_positive_tok",
            "att_sum_negative_tok",
            "att_sum_negative_tok_with_orig",
        ]
    ] = target_df[
        ["att_positive_tok", "att_negative_tok", "att_negative_tok_with_original"]
    ].apply(
        np.vectorize(lambda x: x.sum_to_size(1))
    )

    # normalization
    _max_seq_length = max(df["seq_len_pos"].max(), df["seq_len_neg"].max())
    target_df["att_sum_positive_tok"] /= df["seq_len_pos"] / _max_seq_length
    target_df["att_sum_negative_tok"] /= df["seq_len_neg"] / _max_seq_length
    target_df["att_sum_negative_tok_with_orig"] /= df["seq_len_neg"] / _max_seq_length

    target_df[["special_in_positive", "special_in_negative"]] = df.apply(
        get_attention_on_special_tokens, axis=1, result_type="expand"
    )

    output_dict = {}

    output_dict["seq_len"] = {
        "pos": df["seq_len_pos"].agg(standard_metrics).to_dict(),
        "neg": df["seq_len_neg"].agg(standard_metrics).to_dict(),
    }

    sum_relation = (target_df["att_sum_positive_tok"]) >= (
        target_df["att_sum_negative_tok"]
    )
    num_greater = sum_relation.sum().item()
    num_smaller = len(sum_relation) - num_greater
    output_dict["positive_greater_negative"] = {
        "num_greater": num_greater,
        "num_smaller": num_smaller,
        "percentage": num_greater / (num_greater + num_smaller),
        "seq_len": {
            "pos": df[sum_relation]["seq_len_pos"].agg(standard_metrics).to_dict(),
            "neg": df[sum_relation]["seq_len_neg"].agg(standard_metrics).to_dict(),
        },
    }

    sum_relation = (
        target_df["att_sum_positive_tok"] >= target_df["att_sum_negative_tok_with_orig"]
    )
    num_greater = sum_relation.sum().item()
    num_smaller = len(sum_relation) - num_greater
    output_dict["positive_greater_negative_with_orig"] = (
        {
            "num_greater": num_greater,
            "num_smaller": num_smaller,
            "percentage": num_greater / (num_greater + num_smaller),
            "seq_len": {
                "pos": df[sum_relation]["seq_len_pos"].agg(standard_metrics).to_dict(),
                "neg": df[sum_relation]["seq_len_neg"].agg(standard_metrics).to_dict(),
            },
        }
        if len(df[sum_relation])
        else {}
    )

    att_sum_pos_and_neg_tokens = (
        target_df["att_sum_positive_tok"] + target_df["att_sum_negative_tok"]
    )
    pct_attention_positive = (
        target_df["att_sum_positive_tok"] / att_sum_pos_and_neg_tokens
    )
    pct_attention_negative = (
        target_df["att_sum_negative_tok"] / att_sum_pos_and_neg_tokens
    )

    output_dict["pct_attention_positive"] = pct_attention_positive.agg(
        standard_metrics
    ).to_dict()
    output_dict["pct_attention_negative"] = pct_attention_negative.agg(
        standard_metrics
    ).to_dict()

    output_dict["pct_of_total"] = {
        "positive": (
            target_df["att_sum_positive_tok"]
            / df.attentions_sentence.apply(lambda x: torch.sum(x).item())
        )
        .agg(standard_metrics)
        .to_dict(),
        "negative": (
            target_df["att_sum_negative_tok"]
            / df.attentions_negative.apply(lambda x: torch.sum(x).item())
        )
        .agg(standard_metrics)
        .to_dict(),
        "negative_with_original": (
            target_df["att_sum_negative_tok_with_orig"]
            / df.attentions_negative.apply(lambda x: torch.sum(x).item())
        )
        .agg(standard_metrics)
        .to_dict(),
    }

    output_dict["special_tokens"] = {
        "sum_in_positive": target_df.special_in_positive.apply(lambda x: x.sum().item())
        .agg(standard_metrics)
        .to_dict(),
        "sum_in_negative": target_df.special_in_negative.apply(lambda x: x.sum().item())
        .agg(standard_metrics)
        .to_dict(),
        "pct_in_positive": (
            target_df.special_in_positive.apply(lambda x: x.sum().item())
            / dfs[i].attentions_sentence.apply(lambda x: x.sum().item())
        )
        .agg(standard_metrics)
        .to_dict(),
        "pct_in_negative": (
            target_df.special_in_negative.apply(lambda x: x.sum().item())
            / dfs[i].attentions_negative.apply(lambda x: x.sum().item())
        )
        .agg(standard_metrics)
        .to_dict(),
    }

    # Pre-calculate sum_relation
    sum_relation_np = sum_relation.to_numpy()

    output_dict["special_tokens"]["positive_is_greater"] = (
        {
            "sum_in_positive": target_df.special_in_positive[sum_relation]
            .apply(lambda x: x.sum().item())
            .agg(standard_metrics)
            .to_dict(),
            "sum_in_negative": target_df.special_in_negative[sum_relation]
            .apply(lambda x: x.sum().item())
            .agg(standard_metrics)
            .to_dict(),
        }
        if len(target_df[sum_relation])
        else {}
    )

    def apply_and_agg(series, standard_metrics):
        if series.empty:
            return {metric: None for metric in standard_metrics}
        else:
            return (
                series.apply(lambda x: x.sum().item()).agg(standard_metrics).to_dict()
            )

    output_dict["special_tokens"]["negative_is_greater"] = {
        "sum_in_positive": apply_and_agg(
            target_df.special_in_positive[~sum_relation], standard_metrics
        ),
        "sum_in_negative": apply_and_agg(
            target_df.special_in_negative[~sum_relation], standard_metrics
        ),
    }

    array_positive = target_df.att_sum_positive_tok.to_numpy()
    array_negative = target_df.att_sum_negative_tok.to_numpy()
    array_negative_orig = target_df.att_sum_negative_tok_with_orig.to_numpy()

    output_dict["describe_att_on_positive_tok"] = get_values(
        scipy.stats.describe, array_positive
    )
    output_dict["describe_att_on_negative_tok"] = get_values(
        scipy.stats.describe, array_negative
    )
    output_dict["describe_att_on_negative_tok_with_orig"] = get_values(
        scipy.stats.describe, array_negative_orig
    )

    output_dict["pearsonr"] = get_values(
        scipy.stats.pearsonr, array_positive, array_negative
    )
    output_dict["pearsonr_with_orig"] = get_values(
        scipy.stats.pearsonr, array_positive, array_negative_orig
    )

    output_dict["cov"] = np.cov(array_positive, array_negative, bias=False)[0, 1].item()
    output_dict["cov_with_orig"] = np.cov(
        array_positive, array_negative_orig, bias=False
    )[0, 1].item()

    attention_series_negative = df.apply(split_attentions_by_negative_positions, axis=1)
    attention_series_positive = df.apply(split_attentions_by_positive_positions, axis=1)
    attention_series_negative_orig = df.apply(
        split_attentions_by_negative_with_original_positions, axis=1
    )

    # Pre-calculate common expressions
    sum_attention_negative = np.mean(
        attention_series_negative.apply(lambda item: torch.sum(item[0]).item()).apply(
            np.mean
        )
    )
    sum_attention_positive = np.mean(
        attention_series_positive.apply(lambda item: torch.sum(item[0]).item()).apply(
            np.mean
        )
    )
    sum_attention_negative_orig = np.mean(
        attention_series_negative_orig.apply(
            lambda item: torch.sum(item[0]).item()
        ).apply(np.mean)
    )

    sum_attentions_negative = np.mean(
        df["attentions_negative"].apply(lambda item: torch.sum(item))
    )
    sum_attentions_sentence = np.mean(
        df["attentions_sentence"].apply(lambda item: torch.sum(item))
    )

    output_dict["pct_on_negative_tokens"] = (
        sum_attention_negative / sum_attentions_negative
    ).item()
    output_dict["pct_on_positive_tokens"] = (
        sum_attention_positive / sum_attentions_sentence
    ).item()
    output_dict["pct_on_negative_tokens_with_orig"] = (
        sum_attention_negative_orig / sum_attentions_negative
    ).item()

    output_dict["cos_sim_full_att"] = (
        df["cosine_sim_full_att"].agg(standard_metrics).to_dict()
    )
    output_dict["euc_dist_full_att"] = (
        df["euclidean_distance_full_att"].agg(standard_metrics).to_dict()
    )

    target_df["length_positive"] = target_df["att_positive_tok"].apply(
        lambda item: item.shape[-1]
    )
    target_df["length_negative"] = target_df["att_negative_tok_with_original"].apply(
        lambda item: item.shape[-1]
    )

    entropies_pos_tokens = []
    entropies_neg_tokens = []
    entropies_neg_tokens_orig = []
    max_len = max(
        target_df["length_positive"].max(), target_df["length_negative"].max()
    )
    for length in range(1, max_len + 1):
        tensors_positive = target_df[target_df["length_positive"] == length][
            "att_positive_tok"
        ]
        tensors_negative = target_df[target_df["length_negative"] == length][
            "att_negative_tok"
        ]
        tensors_negative_orig = target_df[target_df["length_negative"] == length][
            "att_negative_tok_with_original"
        ]

        entropy_positive = [
            scipy.stats.entropy(tensor.numpy().ravel()) for tensor in tensors_positive
        ]
        entropy_negative = [
            scipy.stats.entropy(tensor.numpy().ravel()) for tensor in tensors_negative
        ]
        entropy_negative_orig = [
            scipy.stats.entropy(tensor.numpy().ravel())
            for tensor in tensors_negative_orig
        ]

        entropies_pos_tokens.extend(entropy_positive)
        entropies_neg_tokens.extend(entropy_negative)
        entropies_neg_tokens_orig.extend(entropy_negative_orig)

    epsilon = 1e-32  # small constant
    entropies_pos_sentence = [
        scipy.stats.entropy(((item + epsilon) / (item.sum() + epsilon)).numpy().ravel())
        for item in df["attentions_sentence"]
    ]
    entropies_neg_sentence = [
        scipy.stats.entropy(((item + epsilon) / (item.sum() + epsilon)).numpy().ravel())
        for item in df["attentions_negative"]
    ]

    output_dict["entropy"] = {
        "positive_tok": pd.Series(entropies_pos_tokens).agg(standard_metrics).to_dict(),
        "negative_tok": pd.Series(entropies_neg_tokens).agg(standard_metrics).to_dict(),
        "negative_tok_with_orig": pd.Series(entropies_neg_tokens_orig)
        .agg(standard_metrics)
        .to_dict(),
        "positive_sent": pd.Series(entropies_pos_sentence)
        .agg(standard_metrics)
        .to_dict(),
        "negative_sent": pd.Series(entropies_neg_sentence)
        .agg(standard_metrics)
        .to_dict(),
    }

    if "bert" in MODEL_NAME:  # bert-base-cased / roberta-base
        cos_sim = F.cosine_similarity(
            torch.stack(df["pooler_positive"].tolist()).squeeze(1),
            torch.stack(df["pooler_negative"].tolist()).squeeze(1),
            dim=1,
        )
        cos_sim_series = pd.Series(cos_sim)

        euclidean_dist = F.pairwise_distance(
            torch.stack(df["pooler_positive"].tolist()).squeeze(1),
            torch.stack(df["pooler_negative"].tolist()).squeeze(1),
            p=2,
        )
        euclidean_dist_series = pd.Series(euclidean_dist)

        output_dict["pooler_outputs"] = {
            "cos_sim": cos_sim_series.agg(standard_metrics).to_dict(),
            "euc_dist": euclidean_dist_series.agg(standard_metrics).to_dict(),
        }

    output_filename = (
        f"{df.iloc[0,]['dataset_path'].removesuffix('.jsonl')}/results.json"
    )
    target_df["filename"] = output_filename
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    # print(json.dumps(output_df, ensure_ascii=False, indent=2))
    print(f"{output_filename=}")
    outputs.append(output_dict)
    target_dfs.append(target_df)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = sns.color_palette("tab10")
_blue = COLORS[0]
_orange = COLORS[1]
_green = COLORS[2]

for i in range(len(target_dfs)):
    target_df = target_dfs[i]

    plt.figure(figsize=(15, 5), constrained_layout=True)
    _filename = target_df.iloc[0,]["filename"]
    output_filename = f"{os.path.dirname(_filename)}/png/attention_sums_pos_neg.png"

    # plot 1: attention sums on positive/negative tokens
    hist_values = [
        target_df["att_sum_positive_tok"]
        / dfs[i].attentions_sentence.apply(torch.sum).to_list(),
        target_df["att_sum_negative_tok"]
        / dfs[i].attentions_negative.apply(torch.sum).to_list(),
        target_df["att_sum_negative_tok_with_orig"]
        / dfs[i].attentions_negative.apply(torch.sum).to_list(),
    ]
    xlimits = [0, max(hist_value.max() for hist_value in hist_values) * 1.05]
    ylimit = max(
        sns.histplot(hist_value, stat="count").get_yticks().max()
        for hist_value in hist_values
    )
    plt.clf()  # Clear the plot

    sns.set_theme(font_scale=1.25)
    plt.subplot(1, 3, 1)
    plt.xlim(xlimits)
    plt.ylim([0, ylimit])
    sns.histplot(
        hist_values[0],
        color=_blue,
    )
    plt.xlabel("Anteil der Attention\nPositive Tokens")
    plt.ylabel("Anzahl der Beobachtungen")

    plt.subplot(1, 3, 2)
    sns.histplot(
        hist_values[1],
        color=_orange,
    )
    plt.xlim(xlimits)
    plt.ylim([0, ylimit])
    plt.xlabel("Anteil der Attention\nNegative Tokens")
    plt.ylabel("")

    plt.subplot(1, 3, 3)
    sns.histplot(
        hist_values[2],
        color=_green,
    )
    plt.xlim(xlimits)
    plt.ylim([0, ylimit])
    plt.xlabel("Anteil der Attention\nNegative Tokens mit Original")
    plt.ylabel("")

    plt.savefig(output_filename)

    # plot 2: outliers in att_sum_positive_tok
    sns.set_theme(font_scale=2.5)

    for n in range(3):
        for row, att_name in enumerate(
            [
                "att_sum_positive_tok",
                "att_sum_negative_tok",
                "att_sum_negative_tok_with_orig",
            ]
        ):
            normalized_attention = (
                target_df[att_name]
                / torch.sum(torch.tensor(target_df[att_name].values)).numpy()
            )
            target_quantiles = [0.05, 0.5, 0.95]

            for col, quantile in enumerate(target_quantiles):
                quantile_value = normalized_attention.quantile(quantile)

                if quantile == 0.05:
                    indices = normalized_attention[
                        normalized_attention < quantile_value
                    ].index
                elif quantile == 0.5:
                    epsilon = 0.01
                    indices = normalized_attention[
                        (normalized_attention > quantile_value - epsilon)
                        & (normalized_attention < quantile_value + epsilon)
                    ].index
                else:
                    indices = normalized_attention[
                        normalized_attention > quantile_value
                    ].index

                target_idx = indices[n]

                min_value_heatmap = 0
                max_value_heatmap = 1
                min_value_barplot = 0
                max_value_barplot = 1

                for _case in ["positive", "negative"]:
                    output_filename = os.path.join(
                        os.path.dirname(_filename),
                        "png",
                        att_name,
                        f"q{quantile}",
                        f"{n}_{{type}}_{_case}.png",
                    )
                    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                    if "positive" == _case:
                        sentence_name = "tokenized_positive_sentences"
                        attention_name = "attentions_sentence"
                        heatmap_palette = "Blues"
                        barplot_color = _blue
                    else:
                        sentence_name = "tokenized_negative_sentences"
                        attention_name = "attentions_negative"
                        heatmap_palette = "Oranges"
                        barplot_color = _orange

                    quantile_value = normalized_attention.quantile(quantile)

                    tokens = dfs[i].iloc[target_idx][sentence_name]
                    seq_len = max_seq_length(tokens)
                    xlabels = ylabels = remove_padding(tokens)
                    attention = dfs[i].iloc[target_idx][attention_name][
                        :, :, :seq_len, :seq_len
                    ]

                    # heatmap
                    values = torch.sum(attention, dim=(0, 1))
                    values = values / values.sum()
                    if _case == "positive":
                        max_value_heatmap = values.max()

                    fig_subplot = plt.figure(figsize=(10, 10), constrained_layout=True)
                    sns.heatmap(
                        values,
                        xticklabels=xlabels,
                        yticklabels=ylabels,
                        cmap=heatmap_palette,
                        vmin=min_value_heatmap,
                        vmax=max_value_heatmap,
                    )

                    # Save the figure
                    fig_subplot.savefig(output_filename.format(type="heatmap"))
                    plt.close(fig_subplot)

                    # barplot
                    values = torch.sum(attention, dim=(0, 1, 2))
                    values = values / values.sum()
                    if _case == "positive":
                        max_value_barplot = values.max() * 1.05
                    fig_subplot = plt.figure(figsize=(10, 10), constrained_layout=True)
                    ax_subplot = sns.barplot(
                        x=np.arange(len(xlabels)),
                        y=values,
                        color=barplot_color,
                    )
                    ax_subplot.set_ylabel("Anteil der Attention")
                    ax_subplot.set_xticks(np.arange(len(xlabels)))
                    ax_subplot.set_xticklabels(xlabels, rotation=90)
                    ax_subplot.set_ylim([min_value_barplot, max_value_barplot])
                    fig_subplot.savefig(output_filename.format(type="barplot"))
                    plt.close(fig_subplot)

    # plot 3: attention sums on special tokens
    sns.set_theme(font_scale=1.25)

    plt.figure(figsize=(15, 5), constrained_layout=True)
    # plt.suptitle(title)
    output_filename = f"{os.path.dirname(_filename)}/png/attention_special_tokens.png"

    sum_special = target_df[["special_in_positive", "special_in_negative"]].apply(
        np.vectorize(lambda x: x.sum_to_size(1))
    )
    xlimits = [
        min(
            (
                sum_special.special_in_positive
                / dfs[i].attentions_sentence.apply(torch.sum).to_list()
            ).min(),
            (
                sum_special.special_in_negative
                / dfs[i].attentions_negative.apply(torch.sum).to_list()
            ).min(),
        ),
        max(
            (
                sum_special.special_in_positive
                / dfs[i].attentions_sentence.apply(torch.sum).to_list()
            ).max(),
            (
                sum_special.special_in_negative
                / dfs[i].attentions_negative.apply(torch.sum).to_list()
            ).max(),
        ),
    ]

    plt.subplot(1, 2, 1)
    sns.histplot(
        sum_special.special_in_positive
        / dfs[i].attentions_sentence.apply(torch.sum).to_list(),
        color=_blue,
    )
    plt.xlim(xlimits)
    plt.xlabel("Anteil der Attention")
    plt.ylabel("Anzahl der Beobachtungen")

    plt.subplot(1, 2, 2)
    sns.histplot(
        sum_special.special_in_negative
        / dfs[i].attentions_negative.apply(torch.sum).to_list(),
        color=_orange,
    )
    plt.xlim(xlimits)
    plt.xlabel("Anteil der Attention")
    plt.ylabel("")

    plt.savefig(output_filename)
    plt.close()

    special_tokens_quantiles = [0.05, 0.5, 0.95]
    special_tokens_samples = {}

    for quantile in special_tokens_quantiles:
        special_tokens_samples[quantile] = {}

        for special_token_case in ["special_in_positive", "special_in_negative"]:
            quantile_value = sum_special[special_token_case].quantile(quantile)

            if quantile == 0.05:
                epsilon = 0
                indices = []
                while len(indices) == 0:
                    indices = sum_special[
                        sum_special[special_token_case] <= (quantile_value + epsilon)
                    ].index
                    epsilon += 0.01
                print(f"{quantile=}, {special_token_case=}, {epsilon=}")
            elif quantile == 0.5:
                epsilon = 0
                indices = []
                while len(indices) == 0:
                    indices = sum_special[
                        (sum_special[special_token_case] >= (quantile_value - epsilon))
                        & (
                            sum_special[special_token_case]
                            <= (quantile_value + epsilon)
                        )
                    ].index
                    epsilon += 0.05
                print(f"{quantile=}, {special_token_case=}, {epsilon=}")
            else:
                epsilon = 0
                indices = []
                while len(indices) == 0:
                    indices = sum_special[
                        sum_special[special_token_case] >= (quantile_value - epsilon)
                    ].index
                    epsilon += 0.01
                print(f"{quantile=}, {special_token_case=}, {epsilon=}")

            target_idx = indices[0]

            special_tokens_samples[quantile][special_token_case] = dfs[i].loc[
                target_idx
            ]

    # Loop through the quantiles
    for quantile in special_tokens_samples:
        # Loop through the special token cases
        for special_token_case in special_tokens_samples[quantile]:
            # Get the sample for this quantile and special token case
            sample = special_tokens_samples[quantile][special_token_case]
            quantile_value = sum_special[special_token_case].quantile(quantile)

            # Create a new figure
            fig, ax = plt.subplots(figsize=(10, 10))
            # plt.suptitle(f"{title} - {special_token_case} - q{quantile}")

            if "positive" in special_token_case:
                sentence_name = "tokenized_positive_sentences"
                attention_name = "attentions_sentence"
                heatmap_palette = "Blues"
            else:
                sentence_name = "tokenized_negative_sentences"
                attention_name = "attentions_negative"
                heatmap_palette = "Oranges"

            # Get the tokens and attention values for the sentence
            tokens = sample[sentence_name]
            seq_len = max_seq_length(tokens)
            xlabels = ylabels = remove_padding(tokens)
            attention = sample[attention_name][:, :, :seq_len, :seq_len]

            # Get the position of the current axes
            bbox = ax.get_position()

            # Create a new axes for the colorbar to the right of the current axes
            cbar_ax = fig.add_axes([bbox.xmax + 0.005, bbox.ymin, 0.01, bbox.height])

            # Calculate the values to be plotted in the heatmap
            values = torch.sum(attention, dim=(0, 1))
            values = values / values.sum()

            # Create the heatmap
            sns.heatmap(
                values,
                xticklabels=xlabels,
                yticklabels=ylabels,
                ax=ax,
                cbar_ax=cbar_ax,
                cmap=heatmap_palette,
            )

            # Save the figure
            output_filename = f"{os.path.dirname(_filename)}/png/attention_special_tokens_{special_token_case}_quantile_{quantile}.png"
            # plt.tight_layout()
            plt.savefig(output_filename)
            plt.close()
