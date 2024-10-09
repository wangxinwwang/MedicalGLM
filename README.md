# MedicalGLM
## Prepare in advance
### base model: https://huggingface.co/THUDM/chatglm-6b-int4
### reward base model: https://huggingface.co/FacebookAI/xlm-roberta-large/tree/main
## Training the reward model
--model_type roberta
--model_name_or_path
--train_file_dir
--validation_file_dir
--per_device_train_batch_size 1
--per_device_eval_batch_size 1
--do_train
--use_peft True
--seed 42
--max_train_samples 20000
--max_eval_samples 100
--num_train_epochs 6
--learning_rate 0.0001
--warmup_ratio 0.05
--weight_decay 0.001
--logging_strategy steps
--logging_steps 10
--eval_steps 50
--evaluation_strategy steps
--save_steps 500
--save_strategy steps
--save_total_limit 3
--max_source_length 256
--max_target_length 256
--output_dir
--overwrite_output_dir
--ddp_timeout 30000
--logging_first_step True
--target_modules all
--lora_rank 8
--lora_alpha 16
--lora_dropout 0.05
--torch_dtype float32
--device_map auto
--report_to tensorboard
--ddp_find_unused_parameters False
--remove_unused_columns False
--gradient_checkpointing True
--logging_dir ./log
### Merging model
--model_type roberta
--base_model
--peft_model_path
--output_dir
## Fine tuning MedicalGLM
--do_train
--train_file
--validation_file
--test_file
--prompt_column content
--response_column summary
--history_column history
--overwrite_cache
--model_name_or_path
--output_dir
--overwrite_output_dir
--max_source_length 32
--max_target_length 256
--per_device_train_batch_size 1
--per_device_eval_batch_size 1
--gradient_accumulation_steps 1
--predict_with_generate
--max_steps 21060
--logging_steps 50
--save_steps 21060
--learning_rate 0.005
--pre_seq_len 256

 
