# Emotion Classification: BiLSTM vs BERT vs LLM Prompting

A comparative NLP project exploring three approaches to six-class emotion classification: training a bidirectional LSTM from scratch, fine-tuning BERT, and prompting an instruction-tuned large language model.

The project compares classification quality, inference time, and performance with limited labeled training data. It was developed as a collaborative university assignment; the accompanying notebook is provided in its original submitted form.

## Notebook

[View the experiment notebook](notebooke2006d0b77%20%282%29.ipynb)

Place this README beside `notebooke2006d0b77 (2).ipynb` in the repository. The notebook includes code, saved outputs, visualizations, and discussion.

## Dataset

The experiments use the `split` configuration of [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion).

| Split | Examples |
| --- | ---: |
| Training | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |
| Total | 20,000 |

The six labels are **sadness, joy, love, anger, fear, and surprise**. The classes are imbalanced, so macro-F1 is reported alongside accuracy.

## Approaches

### 1. Bidirectional LSTM trained from scratch

- Custom whitespace tokenizer and vocabulary built exclusively from training texts.
- Two bidirectional LSTM layers, 128-dimensional embeddings, and 256 hidden units per direction.
- Approximately 3.32 million trainable parameters.
- Adam optimization, dropout, gradient clipping, learning-rate scheduling, and validation-loss-based early stopping.

### 2. Fine-tuned BERT

- `bert-base-uncased` with a six-class classification head.
- Hugging Face Trainer, dynamic padding, and a maximum sequence length of 128 tokens.
- Three training epochs; best checkpoint selected by validation loss.

### 3. Qwen2.5-3B-Instruct prompting

- `Qwen/Qwen2.5-3B-Instruct` loaded using 4-bit NF4 quantization.
- Zero-shot, 3-shot, 8-shot, and chain-of-thought prompting.
- Few-shot demonstrations drawn from the training split; no model fine-tuning.
- Greedy generation, label parsing, and explicit accounting for unparseable predictions.

The notebook also retrains the LSTM and BERT on an approximately class-balanced **500-example training subset**.

## Recorded results

The following values are taken from the notebook's saved outputs; they have not been independently rerun for this repository description. Each main configuration evaluates the same 2,000-example test split.

| Model / configuration | Test accuracy | Macro-F1 |
| --- | ---: | ---: |
| BiLSTM, full training set | **92.85%** | 0.8796 |
| BERT, full training set* | 92.75% | **0.8862** |
| Qwen2.5-3B, zero-shot | 55.90% | 0.4439 |
| Qwen2.5-3B, 3-shot | **56.55%** | **0.4507** |
| Qwen2.5-3B, 8-shot | 56.50% | 0.4380 |
| Qwen2.5-3B, chain-of-thought | 55.65% | 0.4339 |

\* BERT checkpoint restoration produces parameter-name mismatch warnings. Its recorded scores should be treated as provisional until checkpoint loading is verified and the affected evaluations are repeated.

### Observations from this run

- The BiLSTM achieved the highest recorded accuracy: **92.85%**, or **36.30 percentage points above** the best tested Qwen prompting configuration.
- BERT recorded the highest macro-F1, subject to the checkpoint caveat above.
- Three-shot prompting increased accuracy from **55.90% to 56.55%**, a **0.65 percentage-point** gain over zero-shot in this run.
- Chain-of-thought prompting took substantially longer and did not improve accuracy over zero-shot in this setup.
- On the 500-example subset, recorded accuracy fell to 33.75% for the LSTM and 23.30% for BERT. These observations do not establish that limited data alone caused the drop; the BERT loading issue also needs investigation.

These are comparisons within this experiment, rather than general rankings of the model families or statistically established improvements.

## Evaluation and visualizations

- Accuracy and macro-F1 across the six emotion classes.
- Per-class precision, recall, and F1.
- Confusion matrices and training/validation loss curves.
- Class-frequency and text-length exploration.
- Training and inference timing.
- Full-data versus limited-data comparison.

Unparseable LLM responses are counted as incorrect. Macro-F1 is computed over the six actual emotion classes; confusion matrices also display an unknown-prediction category where applicable.

## Running the notebook

1. Upload the notebook to Kaggle and enable a GPU accelerator and internet access. The saved run used an environment with two NVIDIA Tesla T4 GPUs available; individual components may use different device placement.
2. Open the notebook and run its installation cell, then execute the remaining cells in order.
3. Allow time and disk space for dataset/model downloads, training checkpoints, and generated figures. The saved chain-of-thought evaluation alone took approximately 3.9 hours.

The notebook uses Python, PyTorch, Hugging Face Transformers and Datasets, Accelerate, bitsandbytes, scikit-learn, NumPy, pandas, Matplotlib, and Seaborn. The installation cell installs several of these packages; the original Kaggle environment supplies others. Exact package versions are not pinned, so a new environment may need compatibility adjustments.

## Limitations and reproducibility notes

- **Checkpoint restoration:** BERT outputs include missing `LayerNorm.weight/bias` keys and unexpected `LayerNorm.gamma/beta` keys when loading selected checkpoints. This requires investigation before presenting the BERT scores as verified results.
- **Single-run evidence:** A fixed seed of 42 is used, but repeated-seed experiments and confidence intervals are not included.
- **Prompt coverage:** The 3-shot prompt has examples for sadness, joy, and love only; it does not represent all six labels. The 8-shot prompt includes all six, with additional examples for sadness and joy.
- **Parsing:** Keyword-based extraction can select a mentioned emotion rather than the intended final answer, particularly when chain-of-thought parsing falls back to the general parser.
- **Timing:** Classifiers use batched prediction while LLM prompts are processed sequentially. Timings describe these implementations and are not a controlled comparison of intrinsic model speed.
- **Test-set inspection:** Example generations display test texts and labels. Future prompt tuning should use validation data and reserve test data for final evaluation.
- **Scope:** Results concern this dataset, model selection, and prompt setup. They do not establish that prompting, fine-tuning, or recurrent models are universally preferable.
- **Original discussion:** Some notebook explanations are hypotheses rather than tested causal findings; the qualifications in this README apply to those interpretations.

## Contributors

Collaborative academic project by **Mariam Ayman**, **Nouran Ashraf**, and **Nour Ahmed**, as credited in the notebook. Individual implementation responsibilities are not specified in the submitted notebook.

## Resources

- [Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion)
- [BERT base uncased](https://huggingface.co/google-bert/bert-base-uncased)
- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

Dataset and pretrained-model use remain subject to their respective terms.
