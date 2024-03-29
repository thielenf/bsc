import glob
import numpy as np
import pandas as pd
import os

LAMA_DATASET_PATH = "/media/data/thielen/ba/LAMA_negated"


def wagner_fisher(s: str, t: str):
    """
    Computes the Levenshtein distance between the two strings.  Returns a tuple containing
    the distance itself and also the entire matrix for further processing.
    See: https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
    """
    m, n = len(s), len(t)
    d = np.zeros(shape=(m + 1, n + 1), dtype="int32")

    for i in range(1, m + 1):
        d[i, 0] = i

    for j in range(1, n + 1):
        d[0, j] = j

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                substitutionCost = 0
            else:
                substitutionCost = 1

            d[i, j] = min(
                d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + substitutionCost
            )

    return d[m, n], d


def edit_instructions(s: str, t: str):
    """
    Compute the edit operations required to get from string s to string t
    """
    distance, d = wagner_fisher(s, t)
    m, n = len(s), len(t)
    instructions = []

    while m > 0 or n > 0:
        deletion_score = d[m - 1, n] if m >= 1 else float("inf")
        insertion_score = d[m, n - 1] if n >= 1 else float("inf")
        substitution_or_noop_score = (
            d[m - 1, n - 1] if m >= 1 and n >= 1 else float("inf")
        )
        smallest = min(deletion_score, insertion_score, substitution_or_noop_score)
        if smallest == substitution_or_noop_score:
            if d[m - 1, n - 1] < d[m, n]:
                instructions.append(
                    'substitute "%s" with "%s" at position %d'
                    % (s[m - 1], t[n - 1], n - 1)
                )
            m -= 1
            n -= 1
        elif smallest == deletion_score:
            instructions.append('delete "%s" at position %d' % (s[m - 1], n))
            m -= 1
        elif smallest == insertion_score:
            instructions.append('insert "%s" at position %d' % (t[n - 1], n - 1))
            n -= 1

    if distance != len(instructions):
        raise Exception("Internal error")

    return instructions[::-1]


def get_edit_positions(s, t):
    return list(sorted(set(int(x.split(" ")[-1]) for x in edit_instructions(s, t))))


def unmask_item(item, column_name):
    masked_sentence = item[column_name]
    sub, obj = item["sub_label"], item["obj_label"]
    if type(sub) == float:
        target = obj
    elif type(obj) == float:
        target = sub
    else:
        if masked_sentence.find("[MASK]") < len(masked_sentence) / 2:
            target = sub
        else:
            target = obj

    return masked_sentence.replace("[MASK]", target)


def get_dataset():
    for directory in glob.glob(f"{LAMA_DATASET_PATH}/*"):
        if not os.path.isdir(directory):
            continue
        df = pd.DataFrame()
        for filename in glob.glob(directory + "/*.jsonl"):
            if "_unmasked" in filename:
                continue
            new_df = pd.read_json(filename, lines=True)

            new_df["filename"] = (
                os.path.basename(directory) + "/" + os.path.basename(filename)
            )
            df = pd.concat([df, new_df], ignore_index=True)

        df["dataset"] = os.path.basename(directory)
        df["dataset_path"] = directory

        if "negated" not in df.columns or "masked_sentences" not in df.columns:
            continue

        for col in ["masked_sentences", "negated"]:
            df = df.drop(df[df[col].apply(type) == float].index)

            if df[col].apply(len).value_counts().shape[0] == 1:
                df[col] = df[col].apply(lambda x: x[0])

        df = df.drop_duplicates(subset=["masked_sentences", "negated"]).reset_index(
            drop=True
        )
        df = df.dropna(subset=["negated", "masked_sentences"]).reset_index(drop=True)

        df["unmasked_sentences"] = df.apply(
            lambda item: unmask_item(item, "masked_sentences"),
            axis=1,
        )
        df["unmasked_negated"] = df.apply(
            lambda item: unmask_item(item, "negated"),
            axis=1,
        )
        df["edited_positions_in_negated"] = df.apply(
            lambda s: get_edit_positions(
                s["unmasked_sentences"].split(" "), s["unmasked_negated"].split(" ")
            ),
            axis=1,
        )
        df = df[df["edited_positions_in_negated"].apply(len) > 0].reset_index(drop=True)
        df["edited_tokens_in_negated"] = df.apply(
            lambda s: [
                s["unmasked_negated"].split(" ")[idx]
                for idx in s["edited_positions_in_negated"]
            ],
            axis=1,
        )
        df = df[df["edited_tokens_in_negated"].apply(len) > 0].reset_index(drop=True)

        df["edited_positions_in_unmasked"] = df.apply(
            lambda s: get_edit_positions(
                s["unmasked_negated"].split(" "), s["unmasked_sentences"].split(" ")
            ),
            axis=1,
        )

        df["edited_tokens_in_unmasked"] = df.apply(
            lambda s: [
                s["unmasked_sentences"].split(" ")[idx]
                for idx in s["edited_positions_in_unmasked"]
            ],
            axis=1,
        )

        print(f"Loaded {len(df)} samples from {directory}")
        yield df
