""" please don't judge me for the following code """
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
import yaml
from tqdm import tqdm
import numpy as np
import os
import pandas as pd

import argparse



def try_load(module_name):
    import sys

    return __import__(module_name)


def get_datasets(args: argparse.ArgumentParser):
    if args.dataset_config:
        with open(
            os.path.join("config", "datasets.yaml"), "r", encoding="utf-8"
        ) as ifile:
            datasets = yaml.load(ifile, Loader=yaml.FullLoader)
    else:
        datasets = {
            args.dataset_path: {
                args.dataset_name: {
                    "splits": [args.split],
                    "max_sequence_length": args.max_sequence_length,
                }
            }
        }

    if args.custom_dataset:
        # try to import given path
        try:
            custom_dataset = try_load(args.custom_dataset)
        except ImportError:
            raise ValueError(
                f"Could not import custom dataset from {args.custom_dataset}"
            )

        for df in custom_dataset.get_dataset():
            name = df.iloc[0,]["dataset"]
            df_path = df.iloc[0,]["dataset_path"]

            dataset = Dataset.from_pandas(df)
            kwargs = {
                "splits": [args.split],
                "max_sequence_length": 32,
            }

            dataset_path = os.path.join(df_path, *[str(val) for val in kwargs.values()])
            variant = dataset.map(lambda x: {"inputs": x["unmasked_sentences"]})
            variant_kwargs = kwargs.copy()
            variant_path = f"LAMA_negated/{name}/{args.split}/{kwargs['max_sequence_length']}/variant_00"
            variant_kwargs["dataset_path"] = variant_path
            yield variant, variant_kwargs

            dataset_path = os.path.join(df_path, *[str(val) for val in kwargs.values()])
            variant = dataset.map(lambda x: {"inputs": x["unmasked_negated"]})
            variant_kwargs = kwargs.copy()
            variant_path = f"LAMA_negated/{name}/{args.split}/{kwargs['max_sequence_length']}/variant_01"
            variant_kwargs["dataset_path"] = variant_path
            yield variant, variant_kwargs

        return

    # iterate over every dataset name, path and split combination
    for path, path_params in datasets.items():
        if "negation-dataset" not in path:
            continue
        for name, kwargs in path_params.items():
            for split in kwargs.pop("splits"):
                dataset_path = os.path.join(
                    path, name, split, *[str(val) for val in kwargs.values()]
                )

                dataset = load_dataset(path=path, name=name, split=split)

                if path == "jinaai/negation-dataset":
                    variant = dataset.map(lambda x: {"inputs": x["negative"]})

                    variant_kwargs = kwargs.copy()
                    variant_path = os.path.join(dataset_path, "variant_01")
                    variant_kwargs["dataset_path"] = variant_path
                    yield variant, variant_kwargs

                    dataset = dataset.map(lambda x: {"inputs": x["entailment"]})

                dataset_path = os.path.join(dataset_path, "variant_00")
                kwargs["dataset_path"] = dataset_path

                yield dataset, kwargs


def get_models(args: argparse.ArgumentParser):
    device = args.device

    if args.model_config:
        with open(args.model_config, "r", encoding="utf-8") as ifile:
            model_configs = yaml.load(ifile, Loader=yaml.FullLoader)
        model_names = model_configs.keys()
    else:
        model_names = [args.model]
        model_configs = {
            args.model: {
                "name": os.path.basename(args.model),
                "model_args": args.model_args or {},
                "attention_attribute": args.model_attention_attribute,
            }
        }

    for model_name in model_names:
        model_args = model_configs[model_name]["model_args"]

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, **model_args)

        # Move model to GPU(s)
        if not model_args.get("device_map"):
            model.to(device)
        model.eval()

        yield model, tokenizer, model_configs[model_name]


def parse_arguments():
    # key-value parsing from: https://stackoverflow.com/a/52014520
    def parse_var(s):
        """
        Parse a key, value pair, separated by '='
        That's the reverse of ShellArgs.

        On the command line (argparse) a declaration will typically look like:
            foo=hello
        or
            foo="hello world"
        """
        items = s.split("=")
        key = items[0].strip()  # we remove blanks around keys, as is logical
        if len(items) > 1:
            # rejoin the rest:
            value = "=".join(items[1:])
        return (key, value)

    def parse_vars(items):
        """
        Parse a series of key-value pairs and return a dictionary
        """
        d = {}

        if items:
            for item in items:
                key, value = parse_var(item)
                d[key] = value
        return d

    parser = argparse.ArgumentParser()

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        type=str,
        help="Name of the model to use",
    )
    model_group.add_argument(
        "--model_config",
        type=str,
        help="Name of the model config to use",
    )
    parser.add_argument(
        "--model_args",
        metavar="KEY=VALUE",
        nargs="+",
        help="Extra model args as key-value pairs separated by spaces, e.g. 'key1=value1 key2=value2'",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Dataset path to use, e.g. 'tasksource/bigbench' (overriden by --dataset_config)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name to use, e.g. 'gem' (overriden by --dataset_config)",
    )
    parser.add_argument(
        "--custom_dataset",
        type=str,
        help="Use custom dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Name of the dataset split to use",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        help="Name of the dataset config to use (overrides --dataset)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size to use for inference",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=64,
        help="Maximum sequence length to use for inference",
    )
    parser.add_argument(
        "--model_attention_attribute",
        type=str,
        help="Name of the attribute to access in specified model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for inference",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite existing results",
    )
    parser.add_argument(
        "--dummy_target",
        action="store_true",
        default=False,
        help="Whether to use a dummy target for models with decoder",
    )

    args = parser.parse_args()
    if (
        (not args.dataset_config)
        and (not args.custom_dataset)
        and not all(
            [args.dataset_path != None, args.dataset_name != None, args.split != None]
        )
    ):
        raise ValueError(
            "Either --dataset_config or all of --dataset_path, --dataset_name and --split must be specified"
        )
    if args.model_args:
        args.model_args = parse_vars(args.model_args)
    if not (bool(args.model_config) ^ bool(args.model_attention_attribute)):
        raise ValueError(
            "Either --model-config or --model-attention-attribute must be specified"
        )

    return args


def main():
    # Parse arguments
    args = parse_arguments()

    device = args.device
    num_workers = args.num_workers

    for model, tokenizer, model_config in get_models(args):
        for dataset, dataset_config in get_datasets(args):
            dataset_path = dataset_config["dataset_path"]

            # Create directory for results
            data_path = os.path.join(
                os.sep if args.output_directory.startswith(os.sep) else "",
                *args.output_directory.split(os.sep),
                *model_config["name"].split(os.sep),
                model_config["attention_attribute"],
                *dataset_path.split(os.sep),
            )
            os.makedirs(data_path, exist_ok=True)
            if os.path.exists(
                os.path.join(data_path, "samples.csv")
            ) and os.path.exists(os.path.join(data_path, "attentions.pt")):
                if not args.overwrite:
                    print(
                        f"Skipping {data_path}, since it already exists and not using --overwrite"
                    )
                    continue
                else:
                    print(f"{data_path} exists, but using --overwrite")

            # Define specific configurations for model architectures
            batch_size = args.batch_size
            max_sequence_length = args.max_sequence_length

            # Create dummy target for models with decoder
            dummy_target = (
                torch.ones((batch_size, max_sequence_length), dtype=torch.long).to(
                    device
                )
                if args.dummy_target or model_config.get("dummy_target")
                else None
            )

            # Initialize dataloader
            dataloader = DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=True,
                collate_fn=lambda x: x,  # weird hack, see: https://github.com/pytorch/pytorch/issues/42654#issuecomment-706926806
            )

            run_inference(
                args,
                model,
                model_config,
                tokenizer,
                dataloader,
                data_path,
                dataset_config,
                device,
                dummy_target,
            )

            del dataset, dataloader, dummy_target

        # free memory for next model
        del model, tokenizer
        torch.cuda.empty_cache()


def run_inference(
    args,
    model,
    model_config,
    tokenizer,
    dataloader,
    data_path,
    dataset_config,
    device,
    dummy_target,
):
    # Define specific configurations for model architectures
    max_sequence_length = dataset_config["max_sequence_length"]

    # Initialize results
    sentences = []
    tokens = []
    attention_values = []
    pooler_outputs = []
    print(f"Processing {data_path}")

    # Perform a forward pass with access to hidden states and attentions
    with (
        torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=True
        ),
        torch.no_grad(),
    ):
        for idx, batch in enumerate(tqdm(dataloader)):
            prompts = [
                item.get("inputs_pretokenized") or item.get("inputs") for item in batch
            ]

            # Tokenize and pad the batch to match the model's sequence length
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
            ).to(device)
            input_ids = inputs["input_ids"]

            extra_model_kwargs = model_config.get("extra_model_kwargs", {})
            if "attention_mask" in inputs:
                extra_model_kwargs["attention_mask"] = inputs["attention_mask"]

            if dummy_target is not None:
                # last batch may be different shape, so we have to create new dummy
                if input_ids.shape != dummy_target.shape:
                    dummy_target = torch.ones(input_ids.shape, dtype=torch.long).to(
                        device
                    )
                extra_model_kwargs["decoder_input_ids"] = dummy_target

            # Access attentions by specifying the return_dict argument
            output = model(
                input_ids=input_ids,
                output_attentions=True,
                return_dict=True,
                **extra_model_kwargs,
            )

            # Process the results and store them in the lists
            for i in range(len(prompts)):
                # attentions: <layers>, <batch size>, <heads>, <seq length>, <seq length>
                attentions = getattr(output, model_config["attention_attribute"])

                sentences.append(prompts[i])
                tokens.append(tokenizer.convert_ids_to_tokens(input_ids[i]))
                attention_values.append(
                    [attentions[j][i].cpu() for j in range(len(attentions))]
                )
                if hasattr(output, "pooler_output"):
                    pooler_outputs.append(
                        [output.pooler_output[i].cpu().detach().numpy()]
                    )

    # Create a dataframe to store the results
    df = pd.DataFrame(
        {
            "Sentence": sentences,
            "Tokens": tokens,
        }
    )

    # Save the results
    ixs = np.array_split(df.index, 100)

    # write chunks using tqdm to show progress
    for ix, subset in tqdm(enumerate(ixs), total=100, desc="Writing samples.csv"):
        if ix == 0:
            df.loc[subset].to_csv(
                os.path.join(data_path, "samples.csv"), mode="w", index=True
            )
        else:
            df.loc[subset].to_csv(
                os.path.join(data_path, "samples.csv"),
                header=None,
                mode="a",
                index=True,
            )

    print("Writing attentions.pt")
    torch.save(
        torch.tensor(np.array(attention_values)),
        os.path.join(data_path, "attentions.pt"),
    )

    if len(pooler_outputs) > 0:
        print("Writing pooler_outputs.pt")
        torch.save(
            torch.tensor(np.array(pooler_outputs)),
            os.path.join(data_path, "pooler_outputs.pt"),
        )

    del df, sentences, tokens, attention_values


if __name__ == "__main__":
    main()
