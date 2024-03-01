## Introduction
This is part of dataset and code for paper **Know Your Needs Better: Towards Structured Understanding of Marketer Demands with Analogical Reasoning Augmented LLMs**. This repo aims to display the main prompt content, dataset content, evaluation methods, and fine-tuning code mentioned in our paper. Considering the security of enterprise data, we will release more data content after completing data anonymization in the future. Our project is built on [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), thanks to their awesome work.

**Special Note**: Considering the importance of data security, the tags in the data have been anonymized and rewritten. There might be slight differences compared to the tags that are actually deployed online, but this does not hinder understanding. We hope for your understanding in this matter.

## Test Prompt
In the `row_data` directory, we provide some examples of reasoning library, tag table, train data and test data. Considering the data security, they are anonnymous now. 

`test_prompt.py` provide the different test prompts. The basic instruction comes from the file `test_instruction.txt`. After obtaining the prompt, you can test it by calling the gpt-3.5 turbo API. Please refer to the documentation on the openai official website (https://platform.openai.com/docs/api-reference/chat/object) for the testing script. 

## How to finetune
### Training data
In `sft_data/sft_train_data.json`, we provide two simple train samples corresponding to two training tasks (i.e. predict the answers or predict the reasoning steps). For analogical reasoning based finetune, we will add analogical examples into input, which is similar to `ara_prompt` function in `test_prompt.py`.


### Environment

```bash
git clone https://github.com/wjj0122/ARALLM.git
conda create -n llama_factory python=3.10
conda activate llama_factory
cd ARALLM
pip install -r requirements.txt
```


### Finetune command

#### ChatGLM2-6B-32K
```bash
deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --model_name_or_path YOUR_MODEL_PATH \
    --do_train \
    --dataset DATASET_NAME \
    --template chatglm2 \
    --cutoff_len 4096 \
    --finetuning_type lora \
    --lora_target c_attn \
    --lora_rank 8 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --plot_loss \
    --report_to tensorboard \
    --bf16
```

#### Baichuan2-13B-Chat

```bash
deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --model_name_or_path YOUR_MODEL_PATH \
    --do_train \
    --dataset DATASET_NAME \
    --template baichuan2 \
    --cutoff_len 4096 \
    --finetuning_type lora \
    --lora_target W_pack \
    --lora_rank 8 \
    --output_dir OUTPUT_DIR \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --plot_loss \
    --report_to tensorboard \
    --bf16
```
### Export checkpoint

```bash
python src/export_model.py \
    --model_name_or_path path_to_llama_model \
    --template default \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --export_dir path_to_export
```

### Predict
#### ChatGLM2-6B-32K
```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path YOUR_EXPORT_MODEL_PATH \
    --do_predict \
    --dataset TEST_DATASET \
    --template chatglm2 \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```
#### Baichuan2-13B-Chat
```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path YOUR_EXPORT_MODEL_PATH \
    --do_predict \
    --dataset TEST_DATASET \
    --template baichuan2 \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```
### Evaluate
In  `eval` directory, we provide the evaluation scripts we used for evaluating the result generated from LLMs(ChatGPT or finetuned LLMs). The metrics of structural accuracy and overall accuracy are obtained from `struc_and_overall_eval.py`, while GPTEval are obtained from `gpt4_eval_prompt.py`. The `gpt4_eval_instruction.txt` file provides the scoring example for GPT4 evaluation.
## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

Please follow the model licenses to use the corresponding model weights: [Baichuan](https://huggingface.co/baichuan-inc/Baichuan-13B-Base/resolve/main/Community%20License%20for%20Baichuan-13B%20Model.pdf) / [Baichuan2](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/resolve/main/Community%20License%20for%20Baichuan2%20Model.pdf) / [BLOOM](https://huggingface.co/spaces/bigscience/license) / [ChatGLM3](https://github.com/THUDM/ChatGLM3/blob/main/MODEL_LICENSE) / [Falcon](https://huggingface.co/tiiuae/falcon-180B/blob/main/LICENSE.txt) / [InternLM](https://github.com/InternLM/InternLM#license) / [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) / [LLaMA-2](https://ai.meta.com/llama/license/) / [Mistral](LICENSE) / [Phi-1.5](https://huggingface.co/microsoft/phi-1_5/resolve/main/Research%20License.docx) / [Qwen](https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/LICENSE) / [XVERSE](https://github.com/xverse-ai/XVERSE-13B/blob/main/MODEL_LICENSE.pdf)


