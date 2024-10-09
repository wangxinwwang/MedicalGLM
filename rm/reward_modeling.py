# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import math
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Any, List, Union, Optional, Dict
import json
import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_int8_training
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    PreTrainedTokenizerBase,
    BloomForSequenceClassification,
    LlamaForSequenceClassification,
    LlamaTokenizer,
    BloomTokenizerFast,
    AlbertForSequenceClassification,
    BertForSequenceClassification,
    BertTokenizer,
    AutoTokenizer,
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    RobertaTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    RobertaTokenizerFast,
    AutoModel,
    AutoModelForSeq2SeqLM
)
from transformers.trainer import TRAINING_ARGS_NAME
import torch.nn.functional as F
from trl import AutoModelForSeq2SeqLMWithValueHead

MODEL_CLASSES = {
    "bert": (AutoConfig, BertForSequenceClassification, BertTokenizer),
    "roberta": (AutoConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "albert": (AutoConfig, AlbertForSequenceClassification, AutoTokenizer),
    "chatglm": (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer),
    "bloom": (AutoConfig, BloomForSequenceClassification, BloomTokenizerFast),
    "llama": (AutoConfig, LlamaForSequenceClassification, LlamaTokenizer),
    "auto": (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_type: str = field(
        default=None,
        metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys())}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )

    def __post_init__(self):
        if self.model_type is None:
            raise ValueError(
                "You must specify a valid model_type to run training. Available model types are " + ", ".join(
                    MODEL_CLASSES.keys()))
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The input jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."}, )
    max_source_length: Optional[int] = field(default=256, metadata={"help": "Max length of prompt input text"})
    max_target_length: Optional[int] = field(default=256, metadata={"help": "Max length of output text"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class PeftArguments(TrainingArguments):
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    # fp16 = True


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Here, predictions is rewards_chosen and rewards_rejected.
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    # MSE
    mse = mean_squared_error(labels, preds)
    # MAE
    mae = mean_absolute_error(labels, preds)

    return {"mse": mse, "mae": mae}


#  RewardDataCollatorWithPadding为了处理和准备用于训练或评估的数据
#  特别是在进行奖励模型训练时，处理被选择的（chosen）和被拒绝的（rejected）回答格式的数据。
#  主要包括 tokenizer（用于文本的分词和编码），
#  padding（填充策略），
#  max_length（最大序列长度），
#  pad_to_multiple_of（确保序列长度为某个数的倍数，有助于加速计算），
#  以及 return_tensors（指定返回数据的类型，这里是 "pt"，代表返回PyTorch张量）。
@dataclass
class RewardDataCollatorWithPadding:
    """We need to define a special data collator that batches the data in our chosen vs rejected format"""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        for feature in features:
            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        return batch


#  它是 Trainer 类（来自 Hugging Face Transformers 库）的子类，专门用于训练奖励模型。这个类通过自定义损失计算方法来适应奖励模型的特殊需求

class RewardTrainer(Trainer):
    """
    Trainer for reward models
        Define how to compute the reward loss. Use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    """

    # compute_loss
    # 首先执行模型的前向传播，分别对被选择（chosen）和被拒绝（rejected）的输入进行处理，获取模型输出。
    # 从模型输出中提取 logits（模型对每个输入预测的未归一化分数），并使用 squeeze 方法移除维度为1的轴，以便计算损失。
    # 计算边际排名损失（Margin Ranking Loss），这是一种衡量两组预测结果之间差异的方法，常用于优化排序问题。
    # 提取 embeddings（模型输出的嵌入表示），计算被选择和被拒绝输入之间的对比损失（Contrastive Loss）。这种损失函数通过最小化正样本之间的距离并最大化负样本之间的距离来工作。
    # 最后，将两种损失以某种比例（由 alpha 系数控制）结合起来，计算总损失
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass - get the model outputs for both chosen and rejected
        outputs_chosen = model(input_ids=inputs["input_ids_chosen"],
                               attention_mask=inputs["attention_mask_chosen"])
        outputs_rejected = model(input_ids=inputs["input_ids_rejected"],
                                 attention_mask=inputs["attention_mask_rejected"])

        # Extract logits for margin ranking loss
        logits_chosen = outputs_chosen.logits.squeeze(-1)
        logits_rejected = outputs_rejected.logits.squeeze(-1)

        # Margin Ranking Loss
        margin_ranking_loss = torch.nn.functional.margin_ranking_loss(
            logits_chosen,
            logits_rejected,
            torch.ones_like(logits_chosen),
            margin=1.0
        )
        print("margin_ranking_loss:", margin_ranking_loss)
        # Extract embeddings for contrastive loss
        # embeddings_chosen = outputs_chosen.embeddings  # Ensure your model outputs embeddings
        # embeddings_rejected = outputs_rejected.embeddings

        # Calculate the pairwise distance between embeddings
        # distance = torch.nn.functional.pairwise_distance(embeddings_chosen, embeddings_rejected)
        distance = torch.norm(logits_chosen - logits_rejected, p=2, dim=1)

        # Contrastive Loss
        contrastive_loss = torch.mean(torch.pow(distance, 2))
        print("contrastive_loss:", contrastive_loss)

        # Combine the losses with a balance factor alpha
        alpha = 0.3  # Balance factor, you might want to tune this based on validation performance
        loss = alpha * margin_ranking_loss + (1 - alpha) * contrastive_loss

        if return_outputs:
            return (loss, outputs_chosen, outputs_rejected)
        else:
            return loss

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, margin=1.0):
        device = model.device
        inputs_chosen = {
            "input_ids": inputs["input_ids_chosen"].to(device),
            "attention_mask": inputs["attention_mask_chosen"].to(device),
        }
        outputs_chosen = model(**inputs_chosen)
        rewards_chosen = outputs_chosen.logits.detach()
        inputs_rejected = {
            "input_ids": inputs["input_ids_rejected"].to(device),
            "attention_mask": inputs["attention_mask_rejected"].to(device),
        }
        outputs_rejected = model(**inputs_rejected)
        rewards_rejected = outputs_rejected.logits.detach()
        # Extract logits for margin ranking loss
        logits_chosen = outputs_chosen.logits.squeeze(-1)
        logits_rejected = outputs_rejected.logits.squeeze(-1)
        # Margin Ranking Loss
        margin_ranking_loss = torch.nn.functional.margin_ranking_loss(
            logits_chosen,
            logits_rejected,
            torch.ones_like(logits_chosen),
            margin=1.0
        )
        # print(margin_ranking_loss)
        # Extract embeddings for contrastive loss
        # embeddings_chosen = outputs_chosen.embeddings  # Ensure your model outputs embeddings
        # embeddings_rejected = outputs_rejected.embeddings

        # Calculate the pairwise distance between embeddings
        distance = torch.norm(logits_chosen - logits_rejected, p=2, dim=1)

        # Contrastive Loss
        contrastive_loss = torch.mean(torch.pow(distance, 2))
        # print(contrastive_loss)

        # Combine the losses with a balance factor alpha
        alpha = 0.3  # Balance factor, you might want to tune this based on validation performance
        loss = alpha * margin_ranking_loss + (1 - alpha) * contrastive_loss
        # # 计算对比损失
        # distances = rewards_chosen - rewards_rejected
        # loss = torch.clamp(margin - distances, min=0).mean()

        if prediction_loss_only:
            return (loss.detach(), None, None)
        return (loss.detach(), rewards_chosen, rewards_rejected)


def save_model(output_dir, model, tokenizer, args):
    """Save the model and the tokenizer."""
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, TRAINING_ARGS_NAME))


# CastOutputToFloat 类提供了一个简单而有效的方法来确保模型的输出为 float32 类型，从而增强模型在不同计算环境下的兼容性和数值稳定性
class CastOutputToFloat(torch.nn.Sequential):
    """Cast the output of the model to float"""

    def forward(self, x):
        return super().forward(x).to(torch.float32)


# 计算并打印给定模型中可训练参数的数量、模型中所有参数的总数量以及可训练参数占所有参数总数的百分比。
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# 目的是在使用参数高效微调（PEFT）技术时，找出并列出一个模型中所有适合应用LoRA（Low-Rank Adaptation，低秩适应）技术的线性层的名称。
def find_all_linear_names(peft_model, int4=False, int8=False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'score' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def main():
    # 利用HfArgumentParser解析命令行输入的参数。
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PeftArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    # 为了确保实验的可重复性
    set_seed(training_args.seed)

    # Load model
    if not model_args.model_type:
        raise ValueError("Please specify a model_type, e.g. llama, chatglm, bloom, etc.")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            model_args.device_map = {"": int(os.environ["LOCAL_RANK"]) or 0}
        config = config_class.from_pretrained(
            model_args.model_name_or_path,
            num_labels=1,
            torch_dtype=torch_dtype,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir
        )
        if model_args.model_type in ['bloom', 'llama']:
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                load_in_8bit=model_args.load_in_8bit,
                device_map=model_args.device_map,
                trust_remote_code=model_args.trust_remote_code,
            )
            model.score = CastOutputToFloat(model.score)
        else:
            # model = model_class.from_pretrained(
            #     model_args.model_name_or_path,
            #     config=config,
            #     cache_dir=model_args.cache_dir,
            #     ignore_mismatched_sizes=True
            # )
            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                # load_in_4bit=args.load_in_4bit,
                load_in_8bit=model_args.load_in_8bit,
                # device_map=args.device_map,
                # fp16=True,
                trust_remote_code=model_args.trust_remote_code,
                # peft_config=peft_config if args.use_peft else None,
            )
            model.to(training_args.device).half()
            # if training_args.fp16:
            #     model.half()  # 转换模型到半精度
    else:
        raise ValueError(f"Error, model_name_or_path is None, RM must be loaded from a pre-trained model")

    # Load tokenizer
    if model_args.model_type == "chatglm":
        model_args.use_fast_tokenizer = True
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    # tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    if training_args.use_peft:
        if training_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {training_args.peft_path}")
            model = PeftModel.from_pretrained(model, training_args.peft_path, is_trainable=True)
        else:
            logger.info("Init new peft model")
            target_modules = training_args.target_modules.split(',') if training_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=False, int8=model_args.load_in_8bit)
            modules_to_save = training_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {training_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                target_modules=target_modules,
                inference_mode=False,
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)
        if model_args.load_in_8bit:
            model = prepare_model_for_int8_training(model)
        model.print_trainable_parameters()
    else:
        logger.info("Full parameters training")
        print_trainable_parameters(model)

    # Get reward dataset for tuning the reward model.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {', '.join(train_data_files)}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {', '.join(eval_data_files)}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                'json',
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    full_max_length = data_args.max_source_length + data_args.max_target_length

    def preprocess_reward_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for question, chosen, rejected in zip(examples["question"], examples["response_chosen"],
                                              examples["response_rejected"]):
            # 转换为半精度
            tokenized_chosen = tokenizer("Question: " + question + "\n\nAnswer: " + chosen,
                                         return_tensors="pt", padding=True, truncation=True, max_length=512)
            tokenized_rejected = tokenizer("Question: " + question + "\n\nAnswer: " + rejected,
                                           return_tensors="pt", padding=True, truncation=True, max_length=512)

            if training_args.fp16:
                tokenized_chosen = {k: v.half() for k, v in tokenized_chosen.items()}
                tokenized_rejected = {k: v.half() for k, v in tokenized_rejected.items()}

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"][0])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"][0])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"][0])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"][0])

        return new_examples

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train']
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        # 数据随机化: 通过洗牌来增加模型的泛化能力。
        # 并行预处理: 利用多个进程加速数据预处理。
        # 缓存机制: 可以选择是否从缓存中加载数据，以进一步提高效率。
        # 数据过滤: 确保模型只训练有效（即长度在合适范围内）的序列。
        with training_args.main_process_first(desc="Train dataset tokenization"):
            # 用于随机打乱数据集的顺序
            # 使用 map 函数和 preprocess_reward_function 对数据集进行批量预处理。
            # tokenized_dataset = train_dataset.shuffle().map(
            #     preprocess_reward_function,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     remove_columns=train_dataset.column_names,  # 移除数据集中原有的所有列，因为预处理函数会生成新的列。
            #     load_from_cache_file=not data_args.overwrite_cache,  # 如果设置了 overwrite_cache，则不从缓存文件加载数据。
            #     desc="Running tokenizer on dataset",
            # )
            tokenized_dataset = train_dataset.shuffle().map(
                preprocess_reward_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            # 使用 filter 函数对预处理后的数据集进行过滤。
            train_dataset = tokenized_dataset.filter(
                lambda x: 0 < len(x['input_ids_rejected']) <= full_max_length and 0 < len(
                    x['input_ids_chosen']) <= full_max_length
            )
            logger.debug(f"Num train_samples: {len(train_dataset)}")
            logger.debug("Tokenized training example:")
            logger.debug(tokenizer.decode(train_dataset[0]['input_ids_chosen']))

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            tokenized_dataset = eval_dataset.map(
                preprocess_reward_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            eval_dataset = tokenized_dataset.filter(
                lambda x: 0 < len(x['input_ids_rejected']) <= full_max_length and 0 < len(
                    x['input_ids_chosen']) <= full_max_length
            )
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids_chosen']))

    # Initialize our Trainer
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    model.enable_input_require_grads()
    if torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # 用于计算评估指标的函数
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=full_max_length, padding="max_length"
        ),  # 用于数据整理和填充的数据整理器。
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        logger.debug(f"Train dataloader example: {next(iter(trainer.get_train_dataloader()))}")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        logger.debug(f"Training metrics: {metrics}")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Saving model checkpoint to {training_args.output_dir}")
        save_model(training_args.output_dir, model, tokenizer, training_args)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        logger.debug(f"Eval metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
