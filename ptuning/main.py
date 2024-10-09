#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from trainer_seq2seq import Seq2SeqTrainer

from trainer import Trainer
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)
# 离线状态下可运行
TRANSFORMERS_OFFLINE = 1


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    import torch.distributed as dist
    # 在main函数的开头，加载奖励模型
    quality_model_path = "D:\LLM\MedicalGPT-main\MedicalGPT-main\merged-rmroberta6"
    quality_model = AutoModelForSequenceClassification.from_pretrained(quality_model_path)
    quality_tokenizer = AutoTokenizer.from_pretrained(quality_model_path)
    # 将奖励模型移动到适当的设备上
    quality_model.to(training_args.device)
    quality_model.eval()
    os.environ['MASTER_ADDR'] = 'localhost'

    os.environ['MASTER_PORT'] = '5678'

    dist.init_process_group(backend='gloo', init_method='env://', rank=0, world_size=1)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    training_args.num_train_epochs = 1
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print(raw_datasets)
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # 原版加载模型
    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    # 原版加载模型+输出embedding.weight
    # if model_args.ptuning_checkpoint is not None:
    #     # Evaluation
    #     # Loading extra state dict of prefix encoder
    #     model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    #     # model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    #
    #     # Load the state_dict
    #     prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model-00001-of-00002.bin"))
    #
    #     # Initialize an empty state_dict
    #     new_prefix_state_dict = {}
    #
    #     # Process the prefix state dict
    #     for k, v in prefix_state_dict.items():
    #         if k.startswith("transformer.prefix_encoder."):
    #             new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    #
    #     # Print the embedding.weight if it exists
    #     if "embedding.weight" in new_prefix_state_dict:
    #         print("Embedding weight:", new_prefix_state_dict["embedding.weight"])
    #     else:
    #         print("embedding.weight not found in new_prefix_state_dict")
    #
    #     # Load the combined state_dict into the model
    #     model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    # else:
    #     model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    # 打印键值
    # import torch.nn.functional as F
    #
    # if model_args.ptuning_checkpoint is not None:
    #     # Evaluation
    #     model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    #
    #     prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model-00001-of-00002.bin"))
    #
    #     new_prefix_state_dict = {}
    #
    #     if 'transformer.word_embeddings.weight' in prefix_state_dict:
    #         loaded_weight = prefix_state_dict['transformer.word_embeddings.weight']
    #         expected_shape = model.transformer.prefix_encoder.embedding.weight.shape
    #
    #         print("Loaded embedding.weight shape:", loaded_weight.shape)
    #         print("Model expected embedding.weight shape:", expected_shape)
    #
    #         if loaded_weight.shape != expected_shape:
    #             # Resizing loaded weight to match the expected shape
    #             resized_weight = F.interpolate(loaded_weight.unsqueeze(0), size=expected_shape, mode='nearest').squeeze(
    #                 0)
    #             new_prefix_state_dict['embedding.weight'] = resized_weight
    #             print("Resized embedding.weight to match the model's expected shape.")
    #         else:
    #             new_prefix_state_dict['embedding.weight'] = loaded_weight
    #     else:
    #         raise KeyError("word_embedding.weight not found in the state dict")
    #
    #     for key, value in new_prefix_state_dict.items():
    #         print(f"Key: {key}, Value: {value}")
    #
    #     try:
    #         model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    #     except RuntimeError as e:
    #         print(f"Error in loading state_dict: {e}")
    # else:
    #     model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    # 修改版加载模型+输出embedding.weight
    # if model_args.ptuning_checkpoint is not None:
    #     # Evaluation
    #     # Loading extra state dict of prefix encoder
    #     model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    #
    #     # Initialize an empty state_dict
    #     new_prefix_state_dict = {}
    #
    #     # Iterate over each part of the checkpoint
    #     for part_id in range(1,
    #                          3):  # Assuming two parts: pytorch_model-00001-of-00002.bin, pytorch_model-00002-of-00002.bin
    #         part_file = os.path.join(model_args.ptuning_checkpoint, f"pytorch_model-{part_id:05d}-of-00002.bin")
    #         if os.path.exists(part_file):
    #             prefix_state_dict = torch.load(part_file)
    #             for k, v in prefix_state_dict.items():
    #                 if k.startswith("transformer.prefix_encoder."):
    #                     new_key = k[len("transformer.prefix_encoder."):]
    #                     new_prefix_state_dict[new_key] = v
    #         else:
    #             raise ValueError(f"Checkpoint part {part_id} not found at {part_file}")
    #         # Print the embedding.weight if it exists
    #     if "embedding.weight" in new_prefix_state_dict:
    #         print("Embedding weight:", new_prefix_state_dict["embedding.weight"])
    #     else:
    #         print("embedding.weight not found in new_prefix_state_dict")
    #         # Ensure all necessary keys are present in new_prefix_state_dict
    #     if "embedding.weight" not in new_prefix_state_dict:
    #         # Example of how to initialize a missing key with a default value
    #         # Modify according to your model's structure and requirements
    #         new_prefix_state_dict["embedding.weight"] = torch.randn_like(
    #             model.transformer.prefix_encoder.embedding.weight)
    #     # Load the combined state_dict into the model
    #     # Print the embedding.weight if it exists
    #     #     if "embedding.weight" in new_prefix_state_dict:
    #     #         print("Embedding weight:", new_prefix_state_dict["embedding.weight"])
    #     #     else:
    #     #         print("embedding.weight not found in new_prefix_state_dict")
    #     model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    # else:
    #     model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        # Finetune
        model = model.half()
    # 改的读取模型代码
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # config.pre_seq_len = model_args.pre_seq_len
    # model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    # prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
    #
    # new_prefix_state_dict = {}
    # for k, v in prefix_state_dict.items():
    #     new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    # model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    # model.half().cuda()
    #
    # model = model.eval()

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    prompt_column = 'content'
    response_column = 'summary'
    history_column = data_args.history_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def compute_quality_score(text, device):
        inputs = quality_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = quality_model(**inputs)
        # Assuming the second output logit is the probability for high-quality
        probabilities = torch.softmax(outputs.logits, dim=-1)
        return probabilities[:, 1].item()
    def preprocess_function_eval(examples):
        inputs, targets = [], []
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query = examples[prompt_column][i]
                if history_column is None or len(examples[history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)
                targets.append(examples[response_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["content"] = examples[prompt_column]
        model_inputs["summary"] = examples[response_column]
        return model_inputs

    def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                if history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > data_args.max_source_length - 1:
                    a_ids = a_ids[: data_args.max_source_length - 1]

                if len(b_ids) > data_args.max_target_length - 2:
                    b_ids = b_ids[: data_args.max_target_length - 2]

                # 将 prompt 和 response 拼接起来，并加上特殊的 token
                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = input_ids.index(tokenizer.bos_token_id)  # 152
                mask_position = context_length - 1  # 151
                labels = [-100] * context_length + input_ids[mask_position + 1:]  # 294

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    def print_dataset_example(example):
        print("input_ids", example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("labels", tokenizer.decode(example["labels"]))

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = mi
            n(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        print_dataset_example(train_dataset[0])

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,  # 使用多少进程进行数据预处理。
                remove_columns=column_names,  # 从数据集中移除指定的列。
                load_from_cache_file=not data_args.overwrite_cache,  # 是否从缓存文件中加载数据集。
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(predict_dataset[0])
        print(predict_dataset)

    # Data collator
    # 这段代码的目的是为序列到序列任务设置一个数据整合器，该整合器负责将原始文本数据转化为模型可以处理的格式。
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    # Metric
    # 这个函数的目的是为了在训练或验证过程中提供一个方式来评估模型的性能。使用ROUGE和BLEU分数可以帮助研究者或开发者了解模型生成的文本与真实标签的相似度有多高。
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        print("Preds before decoding:", preds)
        print("Labels:", labels)
        if isinstance(preds, tuple):
            preds = preds[0]
            print("preds", preds)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        print("decoded_preds", decoded_preds)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))
            # Calculate exact match accuracy and F1
            # exact_match = int(pred == label)
            # score_dict["accuracy"].append(exact_match)
            # score_dict["f1"].append(exact_match)

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # Override the decoding parameters of Seq2SeqTrainer
    # 这两段代码确保了在开始模型生成阶段之前，相关的参数已经被正确地设置。
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    #  这段代码的目的是设置用于Beam Search的Beam数量。Beam Search是一个在文本生成任务中常用的技术，可以增加生成文本的质量。
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    # Initialize our Trainer初始化训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        save_prefixencoder=model_args.pre_seq_len is not None,
        quality_model=quality_model,
        quality_tokenizer=quality_tokenizer,
        compute_quality_score=compute_quality_score
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        model.gradient_checkpointing_enable()
        # model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        for name, param in model.named_parameters():
            print(name)
        trainer.save_state()
        torch.save(model.state_dict(), os.path.join(training_args.output_dir, 'pytorch_model.bin'))

    # Evaluation
    results = {}
    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=max_seq_length,
                                   temperature=0.95)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # if training_args.do_predict:
    #     logger.info("*** Predict ***")
    #     predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_seq_length,
    #                                       do_sample=False, top_p=0.7, temperature=0.95)
    #     metrics = predict_results.metrics
    #     max_predict_samples = (
    #         data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    #     )
    #     metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
    #
    #     trainer.log_metrics("predict", metrics)
    #     trainer.save_metrics("predict", metrics)
    #
    #     if trainer.is_world_process_zero():
    #         if training_args.predict_with_generate:
    #             predictions = tokenizer.batch_decode(
    #                 predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #             )
    #             predictions = [pred.strip() for pred in predictions]
    #
    #             # 从 predict_dataset 中获取 content 和 summary 字段
    #             questions = [item['content'] for item in predict_dataset]
    #             responses_chosen = [item['summary'] for item in predict_dataset]
    #
    #             output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
    #             prediction_data = []
    #
    #             for question, response_chosen, response_rejected in zip(questions, responses_chosen, predictions):
    #                 prediction_data.append({
    #                     "question": question,
    #                     "response_chosen": response_chosen,
    #                     "response_rejected": response_rejected
    #                 })
    #
    #             # 保存修改后的数据为 JSON
    #             with open(output_prediction_file, "w", encoding="utf-8") as writer:
    #                 json.dump(prediction_data, writer, ensure_ascii=False, indent=4)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_seq_length,
                                          do_sample=False, top_p=0.7, temperature=0.95)
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
                prediction_data = [{"labels": l, "predict": p} for p, l in zip(predictions, labels)]
                # Save as JSON
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    json.dump(prediction_data, writer, ensure_ascii=False, indent=4)
                # with open(output_prediction_file, "w", encoding="utf-8") as writer:
                #     for p, l in zip(predictions, labels):
                #         res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                #         writer.write(f"{res}\n")
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
