# Reasoning-Aware Abstention for Reducing Hallucinations in Multi-Hop Question Answering

This archive contains the code associated with the MSc Artificial Intelligence dissertation:

**Reasoning-Aware Abstention for Reducing Hallucinations in Multi-Hop Question Answering**

The project investigates whether reliability information extracted from intermediate reasoning can be used to improve selective answering in multi-hop question answering. The experiments study three main stages:

1. estimating answer reliability from reasoning-level, answer-confidence, and final-answer verification signals;
2. using these reliability decisions to construct preference data for Direct Preference Optimisation (DPO);
3. applying verification directly during reasoning through a verifier-in-the-loop approach.

The project is evaluated on HotpotQA and 2WikiMultiHopQA.

---

## Archive Contents

The archive contains the following notebooks:

### `01_iterative_reasoning_generation.ipynb`

Generates intermediate reasoning trajectories and final-answer samples using the base language model.

Main functions include:

- loading and preparing the multi-hop QA data;
- retrieving question-level evidence using BM25;
- generating reasoning incrementally;
- sampling final answers;
- grouping semantically equivalent answers;
- computing answer-confidence information;
- saving generated reasoning and intermediate outputs used by later stages.

---

### `02_reliability_classifier_and_dpo.ipynb`

Implements the post-hoc reliability-estimation pipeline and reliability-guided DPO.

Main functions include:

- retrieving step-specific evidence;
- verifying reasoning steps with an NLI model;
- extracting reasoning-level reliability signals;
- constructing answer-confidence and final-answer verification signals;
- forming the 17-dimensional reliability feature vector;
- training and evaluating the logistic-regression reliability classifier;
- producing low-risk and high-risk decisions;
- constructing answer-versus-abstain preference pairs;
- fine-tuning the language model using DPO;
- evaluating selective-answering performance and decision fidelity.

---

### `03_verifier_in_the_loop.ipynb`

Implements verifier-in-the-loop reasoning.

Main functions include:

- generating candidate reasoning steps;
- retrieving evidence for each candidate;
- verifying candidate steps before they enter the reasoning history;
- rejecting contradicted or conflicting reasoning;
- retaining supported or sufficiently grounded intermediate steps;
- constructing and verifying final-answer claims;
- deciding whether to return an answer or abstain;
- evaluating coverage, selective accuracy, confident error, over-abstention, and THS.

---

### `04_baselines.ipynb`

Implements the selective-answering baselines used for comparison with the proposed approaches.

The evaluated baselines include:

- P(True);
- answer log-probability;
- verbalised confidence;
- semantic entropy;
- R-Tuning;
- CRaFT;
- TruthRL.

The notebook evaluates the methods on the same held-out test questions and using the same correctness-judging procedure as the proposed approaches.

---

### `05_limitation.ipynb`

Contains additional analysis used to investigate limitations and dataset-dependent behaviour, including analyses of reasoning reliability across question types.

---

## Data

The experiments use two publicly available multi-hop question-answering datasets:

- **HotpotQA**
- **2WikiMultiHopQA**

HotpotQA is used in its distractor setting, where the relevant evidence passages are provided together with additional irrelevant passages.

2WikiMultiHopQA contains questions requiring information from multiple pieces of evidence to be combined through different forms of multi-hop reasoning.

For each dataset, a fixed set of **1,000 questions** was used:

- **800 questions** for development and training;
- **200 questions** as a held-out test set.

Where additional parameter selection was required, a fixed **150-question validation subset** was taken from the development portion. The held-out test questions were not used for preference construction or parameter selection.

The original datasets are not included in this archive because they are publicly available and are substantially larger than the submitted project code. Their original publications and dataset details are cited in the dissertation.

---

## Data Processing

The overall processing pipeline is as follows.

1. The question and its provided context passages are loaded from the corresponding dataset.

2. **BM25 retrieval** is used to select evidence relevant to the original question.

3. **Qwen2.5-7B-Instruct** generates reasoning incrementally, one reasoning unit at a time.

4. Multiple final answers are sampled from the resulting reasoning trajectory.

5. Final answers are normalised and grouped according to lexical and semantic equivalence.

6. For reliability estimation, each generated reasoning step is used as a retrieval query to obtain step-specific evidence.

7. A **DeBERTa-v3-large NLI verifier** is used to estimate entailment and contradiction between retrieved evidence and generated claims.

8. The resulting verification scores are converted into reasoning-level reliability features.

9. These are combined with answer-confidence and final-answer verification signals to form the reliability feature vector.

10. A logistic-regression classifier estimates the risk that the generated answer is incorrect.

11. The classifier decisions are used to construct answer-versus-abstain preference pairs for DPO.

12. In the verifier-in-the-loop pipeline, verification is instead applied directly during generation so that unreliable candidate reasoning steps can be rejected before influencing subsequent reasoning.

13. The proposed methods and baselines are evaluated on the same held-out test sets.

Detailed prompts, thresholds, model settings, and decision rules are provided in **Appendix A of the dissertation**.

---

## Main Models and Resources

The experiments use the following main models and resources:

- **Base language model:** `Qwen2.5-7B-Instruct`
- **NLI verifier:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **Sentence-embedding model:** `BAAI/bge-small-en-v1.5`
- **Claim construction and answer-correctness judging:** `GPT-4o-mini`
- **Retrieval:** BM25

The DPO experiments use LoRA fine-tuning with 4-bit NF4 quantisation.

Model checkpoints are not included in the archive and are downloaded from their original providers when required.

API credentials are not included.

---

## Intermediate Files

During development, intermediate experiment files were stored under:

`/content/drive/MyDrive/hedge_run/`

These include generated reasoning trajectories, extracted reliability features, intermediate evaluation outputs, and files passed between different stages of the experimental pipeline.

Large intermediate outputs are not included in the submitted archive.

The notebooks indicate where these files are created, loaded, or saved.

---

## Running the Code

The notebooks are organised according to the main experimental workflow and can be read or run in approximately the following order:

1. `01_iterative_reasoning_generation.ipynb`
2. `02_reliability_classifier_and_dpo.ipynb`
3. `03_verifier_in_the_loop.ipynb`
4. `04_baselines.ipynb`
5. `05_limitation.ipynb`

The first notebook generates the reasoning and answer outputs required for the reliability analysis.

The second notebook extracts the reliability signals, trains the reliability classifier, constructs the DPO preference data, and evaluates reliability-guided DPO.

The third notebook implements the verifier-in-the-loop experiment.

The fourth notebook contains the baseline implementations and comparisons.

The fifth notebook contains additional analysis related to the limitations and question-type behaviour discussed in the dissertation.

Some notebooks use intermediate files produced by earlier stages of the pipeline.

---

## Reproducing the Experimental Outputs

The included notebooks document the code used to generate the main experimental results reported in the dissertation.

Reproduction requires access to:

- the HotpotQA and 2WikiMultiHopQA datasets;
- the external language-model and NLI checkpoints listed above;
- Python packages required by the notebooks;
- an OpenAI API key for the GPT-4o-mini components;
- sufficient GPU resources for language-model inference and DPO fine-tuning.

The experiments involving language-model generation may not be exactly deterministic despite fixed seeds because of sampling and differences between software or hardware environments.

The dissertation therefore provides the fixed dataset splits, model configurations, thresholds, prompts, and main hyperparameters required to understand and reproduce the experimental procedure.

---

## Main Python Dependencies

The notebooks use standard machine-learning and NLP libraries, including packages such as:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `trl`
- `bitsandbytes`
- `scikit-learn`
- `numpy`
- `pandas`
- `rank-bm25`
- `sentence-transformers`
- `openai`

Additional imports required by individual notebooks are shown directly in the corresponding notebook.

---

## Outputs and Evaluation

The main outputs generated by the notebooks include:

- generated reasoning trajectories;
- sampled final answers;
- reasoning-level reliability signals;
- answer-confidence signals;
- final-answer verification signals;
- reliability-classifier predictions;
- DPO preference pairs;
- reliability-guided DPO outputs;
- verifier-in-the-loop outputs;
- baseline outputs;
- selective-answering evaluation metrics.

The main reported metrics include:

- AUROC;
- coverage;
- selective accuracy;
- confident error;
- over-abstention;
- Truthful Helpfulness Score (THS);
- DPO decision fidelity.

The exact experimental configuration and the final reported values are provided in the dissertation.

---

## Files Not Included

In accordance with the project-submission guidance, the archive does not include large files that are not necessary for examining the submitted work.

In particular, the following are not included:

- complete copies of the public datasets;
- pretrained model checkpoints;
- DPO model checkpoints;
- model caches;
- large generated-output files;
- large intermediate experiment files;
- API credentials.

These can be regenerated or obtained from their original sources where required.

The submitted archive contains the code and documentation needed to inspect the implementation and understand how the experimental pipeline and reported outputs were produced.

---

## Dissertation

The accompanying dissertation contains the full description of the methodology, experimental setup, results, limitations, prompts, thresholds, and implementation details.

In particular, **Appendix A: Implementation and Reproducibility Details** provides additional information required to interpret and reproduce the experiments.