# Can Small Language Models Plan Hospital Workflows Under Operational Constraints?

## Abstract

This project evaluates whether constraint-aware prompting improves a small language model's ability to produce feasible hospital administrative workflow plans under operational constraints. The experiment uses Qwen3-4B and 15 fictional scenarios covering power outages, software failures, limited computer availability, staff guidance, incomplete records, and combined constraints.

Each scenario was evaluated under two prompting conditions: standard prompting and constraint-aware prompting, producing 30 model responses. Responses were manually scored across four dimensions: constraint adherence, feasibility, assumption control, and adaptation specificity. Each dimension received a score from 0 to 2, for a maximum total score of 8.

In this preliminary evaluation, constraint-aware prompting achieved a mean total score of 6.87, compared with 6.53 for standard prompting. No constraint violations were recorded in the 15 constraint-aware responses, while 4 of the 15 standard responses were recorded as violations. These results are descriptive and limited to the scenarios, model, prompts, and scoring procedure used in this experiment.

## 1. Motivation

Hospitals operate under practical constraints that can interrupt administrative workflows. Examples include power outages, unavailable computers, software failures, incomplete digital records, and the need for staff to follow established procedures.

Language models may help staff draft operational plans, identify dependencies, and propose alternative workflows. However, a plan that ignores a critical constraint can be impractical or create additional operational risks.

This study investigates whether explicitly instructing a model to identify constraints, check proposed actions, and revise conflicting steps improves the quality of its workflow plans.

The scenarios are fictional and designed for research. They do not contain real patient records or confidential hospital information.

## 2. Research Questions

**RQ1.** Does constraint-aware prompting improve the overall quality of hospital administrative workflow plans compared with standard prompting?

**RQ2.** Does constraint-aware prompting reduce observed violations of explicitly stated operational constraints?

**RQ3.** How do the two prompting conditions compare across constraint adherence, feasibility, assumption control, and adaptation specificity?

## 3. Model and Reproducibility

- **Model:** `Qwen/Qwen3-4B`
- **Model repository:** https://huggingface.co/Qwen/Qwen3-4B
- **Resolved model revision:** `1cfa9a7208912126459214e8b04321603b3df60c`
- **License metadata:** `apache-2.0`
- **Inference framework:** Hugging Face Transformers with PyTorch
- **Chat template:** `tokenizer.apply_chat_template(...)`
- **Thinking mode:** Disabled using `enable_thinking=False`
- **Maximum new tokens:** 300
- **Decoding:** Greedy decoding with `do_sample=False`
- **Padding token:** `tokenizer.eos_token_id`
- **Inference context:** `torch.inference_mode()`

The revision above is the cached snapshot revision reported by the experiment environment. Recording this revision makes the model artifact more reproducible than recording the model name alone.

The generation function used in the experiment was:

```python
def generate_response(
    user_prompt,
    system_prompt,
    max_new_tokens=300
):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output[
        0,
        inputs["input_ids"].shape[-1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()
```

No explicit temperature, top-p, or top-k value was passed to `model.generate()` in this function. The experiment therefore reports the generation arguments actually shown in the implementation rather than assuming additional settings.

## 4. Evaluation Dataset

The evaluation dataset contains 15 fictional hospital operations scenarios.

| Category | Scenario IDs | Count |
|---|---|---:|
| Power outages | PWR01–PWR03 | 3 |
| Software failures | SW01–SW03 | 3 |
| Computer availability | CMP01–CMP03 | 3 |
| Staff guidance | STF01–STF03 | 3 |
| Incomplete records | REC01–REC02 | 2 |
| Combined constraints | MIX01 | 1 |
| **Total** | | **15** |

The scenarios represent operational planning challenges rather than clinical diagnosis or treatment tasks. The model is instructed to avoid making clinical decisions.

Dataset file: `evaluation_dataset.jsonl`

## 5. Experimental Methodology

### 5.1 Prompting conditions

Each of the 15 scenarios was evaluated using two prompting conditions.

**Standard prompting**

The model was asked to propose a practical workflow, identify immediate and longer-term actions, describe dependencies, and acknowledge unknown information. It was also instructed to avoid clinical decisions.

**Constraint-aware prompting**

The model was instructed to:

1. Identify the stated operational constraints.
2. Identify missing information.
3. Draft a workflow plan.
4. Check each proposed action against the constraints.
5. Revise any actions that conflict with the constraints.
6. Explain relevant risks and limitations.

### 5.2 Experimental design

Each scenario was run under both prompting conditions, producing 30 responses in total:

- 15 standard-prompt responses
- 15 constraint-aware responses

The same model and generation function were used for both conditions. The comparison is paired by scenario.

### 5.3 Manual scoring rubric

Each response was scored on four dimensions, each from 0 to 2.

| Dimension | What it evaluates |
|---|---|
| Constraint adherence | Whether the plan respects the scenario's stated constraints |
| Feasibility | Whether the proposed actions are practically achievable |
| Assumption control | Whether unsupported assumptions are avoided or clearly identified |
| Adaptation specificity | Whether the proposed adaptations respond specifically to the scenario |

The maximum total score is 8 per response.

A separate binary field, `constraint_violation`, records whether an explicit operational constraint violation was identified:

- `1` — violation recorded
- `0` — no violation recorded
- `NA` — unclear or not assessable

The violation measure is separate from the total quality score.

## 6. Results

### 6.1 Mean quality scores

| Metric | Constraint-aware | Standard |
|---|---:|---:|
| Mean total score (maximum 8) | 6.87 | 6.53 |
| Mean constraint adherence (maximum 2) | 1.87 | 1.67 |
| Mean feasibility (maximum 2) | 1.53 | 1.33 |
| Mean assumption control (maximum 2) | 1.47 | 1.33 |
| Mean adaptation specificity (maximum 2) | 2.00 | 1.93 |
| Number of responses | 15 | 15 |

Constraint-aware prompting had a mean total score 0.33 points higher than standard prompting.

### 6.2 Recorded constraint violations

| Prompting condition | Recorded violations | Responses | Rate |
|---|---:|---:|---:|
| Constraint-aware | 0 | 15 | 0.0% |
| Standard | 4 | 15 | 26.7% |

The four standard-prompt responses marked as violations were associated with scenarios `CMP01`, `CMP02`, `PWR01`, and `PWR02`.

These figures describe the manually evaluated responses in this experiment. They do not establish that constraint-aware prompting eliminates operational violations in real-world hospital settings.

### 6.3 Paired scenario comparison

For each scenario, the constraint-aware score was compared with the standard-prompt score.

- Constraint-aware score higher: 4 scenarios
- Standard score higher: 1 scenario
- Equal scores: 10 scenarios
- Mean paired score difference (constraint-aware minus standard): +0.33 points

The paired results suggest a modest improvement in overall scores for this dataset, with most scenarios receiving equal scores under both conditions.

## 7. Discussion

The preliminary results are consistent with the possibility that explicit constraint-checking instructions can improve a model's operational planning responses.

Three observations merit further investigation:

1. **Constraint adherence:** The constraint-aware condition achieved a higher mean adherence score.
2. **Recorded violations:** No violations were recorded in the constraint-aware responses, compared with four in the standard condition.
3. **Overall score:** The mean total score was higher under constraint-aware prompting, although the absolute difference was modest.

The findings should be interpreted cautiously. The evaluation includes only one model, 15 fictional scenarios, two prompting conditions, and manually assigned scores. The results do not establish statistical significance, broad generalizability, or clinical or operational safety.

Before making stronger claims, the violation labels should be checked against the original scenario constraints and the exact model responses. Representative examples should be documented so readers can understand why each response was classified as a violation.

## 8. Proposed Improvement

A potential next step is to compare the existing prompting conditions with a structured output format requiring the model to report:

- Identified constraints
- Missing information
- Proposed actions
- Dependencies and responsible roles
- Feasibility checks
- Conflicts and revised actions
- Escalation requirements
- Remaining uncertainties

Future work could also include independent evaluators, a documented adjudication process, additional scenarios, repeated runs where appropriate, and comparisons with other small language models.

These are proposed extensions, not completed experiments.

## 9. Limitations

This study has several limitations:

- **Small dataset:** Only 15 scenarios were evaluated.
- **Single model:** The experiment used Qwen3-4B only.
- **Prompt sensitivity:** Results may depend on the exact wording and structure of the prompts.
- **Manual scoring:** Scores and violation labels may be affected by evaluator judgment.
- **Limited validation:** The experiment did not test the plans in a live hospital environment.
- **Fictional scenarios:** Results may not represent the full complexity of actual hospital operations.
- **Descriptive comparison:** The reported summary does not establish statistical significance.
- **No clinical validation:** The model was not evaluated for clinical decision-making or patient-care recommendations.

Accordingly, the results should be treated as an exploratory evaluation of prompting strategies, not evidence that the model is ready for autonomous hospital workflow management.

## 10. Ethical Considerations

All evaluation scenarios are fictional. No real patient records or confidential hospital data were used in the scenario design described here.

The experiment concerns administrative and operational planning, not diagnosis, prescribing, triage, or other clinical decisions. Any future application in a hospital would require appropriate human oversight, privacy and security controls, local workflow validation, and institutional approval.

Model-generated plans should not be executed automatically based solely on this evaluation.

## 11. Reproducibility and Repository Contents

The repository should contain the following artifacts once each file has been verified and added:

```text
.
├── README.md
├── Hospital_Operational_Constraints_Evaluation.ipynb
├── evaluation_dataset.jsonl
├── model_outputs.csv
├── manual_scores_completed.csv
├── summary_metrics.csv
└── requirements.txt
```

The notebook should document scenario loading, prompt construction, model inference, scoring-data preparation, metric calculations, and result generation.

The output CSV should preserve the generated responses and enough scenario and condition information to link each response to its evaluation record. The completed scoring CSV should preserve the manual scores and violation labels. The summary metrics file should be generated from those scoring records.

Only list files in the published repository after confirming that they exist and have been uploaded. Do not publish access tokens, private credentials, or confidential data.

### Reproduction notes

The reported model revision is:

`1cfa9a7208912126459214e8b04321603b3df60c`

The recorded generation settings are `max_new_tokens=300`, `do_sample=False`, `pad_token_id=tokenizer.eos_token_id`, and `enable_thinking=False`.

Python, PyTorch, Transformers, hardware, and other dependency versions should be recorded from the actual experiment environment before publication. They are intentionally not specified here because their exact values have not yet been provided.

## 12. Conclusion

In this small, paired evaluation, constraint-aware prompting achieved a mean total score of 6.87 out of 8, compared with 6.53 for standard prompting. It also had no recorded constraint violations among 15 responses, while four violations were recorded in the standard condition.

These preliminary results motivate further investigation into explicit constraint-checking prompts for small language models used in operational planning. Larger evaluations, independent scoring, and real-world validation would be necessary before drawing stronger conclusions about reliability or deployment readiness.