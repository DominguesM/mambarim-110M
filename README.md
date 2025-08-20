---
library_name: transformers
language:
    - pt
license: cc-by-4.0
tags:
    - text-generation
    - pytorch
    - LLM
    - Portuguese
    - mamba
datasets:
    - nicholasKluge/Pt-Corpus-Instruct-tokenized-large
inference:
    parameters:
        repetition_penalty: 1.2
        temperature: 0.8
        top_k: 50
        top_p: 0.85
        max_new_tokens: 150
widget:
    - text: "O Natal é uma"
      example_title: Exemplo
    - text: "A muitos anos atrás, em uma galáxia muito distante, vivia uma raça de"
      example_title: Exemplo
    - text: "Em meio a um escândalo, a frente parlamentar pediu ao Senador Silva para"
      example_title: Exemplo
pipeline_tag: text-generation
---

# Mambarim-110M

<p align="center">
  <img width="350" alt="Camarim Logo" src="https://raw.githubusercontent.com/DominguesM/mambarim-110M/main/assets/mambarim-bg.png">
</p>

</br>

## Model Summary

**Mambarim-110M** is a pioneering 110-million-parameter language model for Portuguese, built upon the **Mamba architecture**. Unlike traditional Transformer models that rely on quadratic self-attention, Mamba is a **State-Space Model (SSM)** that processes sequences with linear complexity.

This design choice leads to significantly faster inference and reduced memory consumption, especially for long sequences. Mamba employs a selection mechanism that allows it to effectively focus on relevant information in the context, making it a powerful and efficient alternative to Transformers. Mambarim-110M is one of the first Mamba-based models developed specifically for the Portuguese language.

## Details

- **Architecture:** a Mamba model pre-trained via causal language modeling
- **Size:** 119,930,880 parameters
- **Context length:** 2048 tokens
- **Dataset:** [Pt-Corpus-Instruct-tokenized-large](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct-tokenized-large) (6.2B tokens)
- **Language:** Portuguese
- **Number of steps:** 758,423

### Training & Reproducibility

This model was trained to be fully open and reproducible. You can find all the resources used below:

- **Source Code:** [GitHub Repository](https://github.com/DominguesM/mambarim-110M/)
- **Training Notebook:** [Open in Colab](https://githubtocolab.com/DominguesM/mambarim-110M/blob/main/MAMBARIM_110M.ipynb)
- **Training Metrics:** [View on Weights & Biases](https://wandb.ai/dominguesm/canarim-mamba-110m?nw=nwuserdominguesm)

## Intended Uses

This model is intended for a variety of text generation tasks in Portuguese. Given its size, it is particularly well-suited for environments with limited computational resources.

- **General-Purpose Text Generation:** The model can be used for creative writing, continuing a story, or generating text based on a prompt.
- **Research and Education:** As one of the first Portuguese Mamba models, it serves as an excellent resource for researchers studying State-Space Models, computational efficiency in LLMs, and NLP for non-English languages. Its smaller size also makes it an accessible tool for educational purposes.
- **Fine-tuning Base:** The model can be fine-tuned on specific datasets to create more specialized models for tasks like simple chatbots, content creation aids, or domain-specific text generation.

## Out-of-scope Use

The model is not intended for use in critical applications without comprehensive testing and fine-tuning. Users should be aware of the following limitations:

- **Factual Accuracy:** This model is not a knowledge base and can generate incorrect or fabricated information ("hallucinate"). It should not be used as a source of truth.
- **High-Stakes Decisions:** Do not use this model for making important decisions in domains such as medical, legal, or financial advice, as its outputs may be unreliable.
- **Bias and Safety:** The model was trained on a large corpus of public data from the internet and may reflect societal biases present in that data. It can generate content that is biased, offensive, or otherwise harmful.

## Basic usage

You need to install `transformers` from `main` until `transformers>=4.39.0` is released.

```bash
pip install git+https://github.com/huggingface/transformers@main
```

We also recommend you to install both `causal_conv_1d` and `mamba-ssm` using:

```bash
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```

You can use the classic `generate` API:

```python
>>> from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
>>> import torch
>>> tokenizer = AutoTokenizer.from_pretrained("dominguesm/mambarim-110m")
>>> model = MambaForCausalLM.from_pretrained("dominguesm/mambarim-110m")
>>> input_ids = tokenizer("O Natal é uma", return_tensors="pt")["input_ids"]
>>> out = model.generate(
    input_ids,
    repetition_penalty=1.2,
    temperature=0.8,
    top_k=50,
    top_p=0.85,
    do_sample=True,
    max_new_tokens=10
)
>>> print(tokenizer.batch_decode(out))
["<s> O Natal é uma data em que as pessoas passam horas de lazer e"]
```

## Benchmarks

Evaluations on Brazilian Portuguese benchmarks were performed using a [Portuguese implementation of the EleutherAI LM Evaluation Harness](https://github.com/eduagarcia/lm-evaluation-harness-pt) (created by [Eduardo Garcia](https://github.com/eduagarcia/lm-evaluation-harness-pt)).

Detailed results can be found [here](https://huggingface.co/datasets/eduagarcia-temp/llm_pt_leaderboard_raw_results/tree/main/dominguesm/mambarim-110m)

| Model                                                                                     | **Average** | ENEM  | BLUEX | OAB Exams | ASSIN2 RTE | ASSIN2 STS | FAQNAD NLI | HateBR | PT Hate Speech | tweetSentBR | **Architecture**     |
| ----------------------------------------------------------------------------------------- | ----------- | ----- | ----- | --------- | ---------- | ---------- | ---------- | ------ | -------------- | ----------- | -------------------- |
| [TeenyTinyLlama-460m](https://huggingface.co/nicholasKluge/TeenyTinyLlama-460m)           | 28.86       | 20.15 | 25.73 | 27.02     | 53.61      | 13         | 46.41      | 33.59  | 22.99          | 17.28       | LlamaForCausalLM     |
| [TeenyTinyLlama-160m](https://huggingface.co/nicholasKluge/TeenyTinyLlama-160m)           | 28.2        | 19.24 | 23.09 | 22.37     | 53.97      | 0.24       | 43.97      | 36.92  | 42.63          | 11.39       | LlamaForCausalLM     |
| [MulaBR/Mula-4x160-v0.1](https://huggingface.co/MulaBR/Mula-4x160-v0.1)                   | 26.24       | 21.34 | 25.17 | 25.06     | 33.57      | 11.35      | 43.97      | 41.5   | 22.99          | 11.24       | MixtralForCausalLM   |
| [TeenyTinyLlama-460m-Chat](https://huggingface.co/nicholasKluge/TeenyTinyLlama-460m-Chat) | 25.49       | 20.29 | 25.45 | 26.74     | 43.77      | 4.52       | 34         | 33.49  | 22.99          | 18.13       | LlamaForCausalLM     |
| [**Mambarim-110M**](https://huggingface.co/dominguesm/mambarim-110m)                      | **14.16**   | 18.4  | 10.57 | 21.87     | 16.09      | 1.89       | 9.29       | 15.75  | 17.77          | 15.79       | **MambaForCausalLM** |
| [GloriaTA-3B](https://huggingface.co/NOVA-vision-language/GlorIA-1.3B)                    | 4.09        | 1.89  | 3.2   | 5.19      | 0          | 2.32       | 0.26       | 0.28   | 23.52          | 0.19        | GPTNeoForCausalLM    |
