import argparse
import json
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from roformer import RoFormerForSequenceClassification

task_to_keys = {
    "iflytek": ("sentence", None),
    "tnews": ("sentence", None),
    "afqmc": ("sentence1", "sentence2"),
    "cmnli": ("sentence1", "sentence2"),
    "ocnli": ("sentence1", "sentence2"),
    "cluewsc2020": ("text", None),
    "csl": ("keyword", "abst"),
}
# 11
task_to_outputfile11 = {
    "iflytek": "iflytek_predict.json",
    "tnews": "tnews11_predict.json",
    "afqmc": "afqmc_predict.json",
    "cmnli": "cmnli_predict.json",
    "ocnli": "ocnli_50k_predict.json",
    "cluewsc2020": "cluewsc11_predict.json",
    "csl": "csl_predict.json",
}
# 1.0
task_to_outputfile10 = {
    "tnews": "tnews10_predict.json",
    "cluewsc2020": "cluewsc10_predict.json",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a transformers model on a text classification task"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the CLUE task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1.0,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None:
        raise ValueError("task_name should not be none.")

    return args


def predict():
    args = parse_args()
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = RoFormerForSequenceClassification.from_pretrained(args.model_name_or_path)
    model.eval()

    all_eval_datasets = []
    all_file_names = []
    if args.task_name in task_to_outputfile10.keys():
        all_eval_datasets.append(
            load_dataset(
                "clue_10.py", args.task_name, cache_dir="./clue_caches_10", split="test"
            )
        )
        all_file_names.append(task_to_outputfile10[args.task_name])

    if args.task_name in task_to_outputfile11.keys():
        all_eval_datasets.append(
            load_dataset(
                "clue_11.py", args.task_name, cache_dir="./clue_caches", split="test"
            )
        )
        all_file_names.append(task_to_outputfile11[args.task_name])

    for raw_test_dataset, file in zip(all_eval_datasets, all_file_names):
        os.makedirs("results", exist_ok=True)
        out_file = f"results/{file}"
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
        int2str = raw_test_dataset.features["label"].int2str
        padding = "max_length" if args.pad_to_max_length else False

        def preprocess_test_function(examples):
            # Tokenize the texts
            if sentence1_key == "keyword":
                k1 = ["；".join(l) for l in examples[sentence1_key]]
            else:
                k1 = examples[sentence1_key]
            texts = (k1,) if sentence2_key is None else (k1, examples[sentence2_key])
            result = tokenizer(
                *texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                return_token_type_ids=False,
            )
            return result

        with accelerator.main_process_first():
            processed_test_dataset = raw_test_dataset.map(
                preprocess_test_function,
                batched=True,
                remove_columns=raw_test_dataset.column_names,
                desc="Running tokenizer on test dataset",
            )
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
        test_dataloader = DataLoader(
            processed_test_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )
        model, test_dataloader = accelerator.prepare(model, test_dataloader)

        samples_seen = 0
        all_predictions = []

        with torch.no_grad():
            for step, batch in enumerate(tqdm(test_dataloader)):
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
                predictions = accelerator.gather(predictions)
                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(test_dataloader):
                        predictions = predictions[
                            : len(test_dataloader.dataset) - samples_seen
                        ]
                        references = references[
                            : len(test_dataloader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += references.shape[0]
                all_predictions.extend(int2str(predictions))

        with open(out_file, "w") as fw:
            for idx, pred in zip(raw_test_dataset["idx"], all_predictions):
                l = json.dumps({"id": str(idx), "label": pred})
                fw.write(l + "\n")
        fw.close()


if __name__ == "__main__":
    predict()
