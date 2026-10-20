# Qualitative Review of Model Outputs

## 1. Purpose

This document reviews selected outputs from an evaluation of `Qwen/Qwen3-4B` on fictional hospital administrative workflow scenarios involving operational constraints.

The review compares two prompting conditions:

- **Standard:** The model receives the scenario and is asked to recommend a practical plan.
- **Constraint-aware:** The model is explicitly instructed to identify constraints and missing information before drafting its plan.

The purpose is to examine whether the responses respect the stated operational constraints, not to assess clinical decision-making or establish real-world hospital safety.

## 2. Review Method

Four scenarios were selected for qualitative review because their standard-prompt responses were recorded as constraint violations during manual evaluation:

- `PWR01` — Power interruptions and record preservation
- `PWR02` — Intermittent electricity and no immediate infrastructure budget
- `CMP01` — Limited shared computers and no new purchases this quarter
- `CMP02` — Limited functioning workstations and competing staff responsibilities

The review considers whether the recommendations are compatible with the stated constraints and whether the alternative response offers a more feasible approach.

**Important limitation:** The supplied responses are truncated. Conclusions below are limited to the visible text and the recorded manual scores. Where a visible excerpt does not establish a violation by itself, that uncertainty is stated explicitly.

## 3. Scenario Reviews

### PWR01 — Routine Administrative Records During Power Interruptions

**Explicit constraints**
- Electricity may be unavailable for several hours.
- Staff must preserve records and avoid duplicate entries.

**Standard response**

The standard response recommends:

> “Ensure that critical systems ... have access to backup generators or battery-powered devices.”

It also recommends testing backup power to sustain operations for at least 48 hours.

**Constraint assessment**

The recommendation assumes backup power can be made available and introduces a 48-hour target without evidence that the facility has the equipment, budget, or operational capacity to support it. The visible response does not establish that these resources are available.

However, the scenario does not explicitly prohibit existing backup power or require a zero-cost solution. Therefore, the recommendation is potentially assumption-heavy rather than conclusively infeasible based on the excerpt alone.

**Constraint-aware response**

The constraint-aware response first identifies the need to determine which backup power sources and downtime procedures are available. It recommends preparation for manual recordkeeping and duplicate prevention, including unique identifiers and cross-referencing.

**Assessment**

The constraint-aware response makes the resource uncertainty explicit before recommending solutions. This is a more cautious planning approach, although the visible excerpt also mentions backup systems without confirming their availability.

**Review finding:** The constraint-aware response demonstrates better explicit handling of missing information. The severity of the standard response's violation should be verified against the full response and the original scoring criteria.

### PWR02 — Administrative Work Without Immediate Infrastructure Upgrades

**Explicit constraints**
- The facility has intermittent electricity.
- There is no budget for immediate infrastructure upgrades.

**Standard response**

The standard response recommends:

> “Identify and secure a reliable backup power source, such as a generator or battery-powered system.”

It also proposes a work hub and a schedule for administrative tasks during outages.

**Constraint assessment**

Recommending that the facility secure a generator or battery-powered system may require spending or procurement that conflicts with the explicit lack of budget for immediate infrastructure upgrades. The response itself acknowledges that a generator or alternative power source must be available, but it does not establish that one exists.

This is a plausible constraint violation because the proposed action may depend on an unavailable investment.

**Constraint-aware response**

The constraint-aware response explicitly recognizes the absence of a budget for infrastructure upgrades and proposes an “Outage-Ready Administrative Workflow” using available resources and manual methods.

It recommends assessing task criticality and identifying manual alternatives.

**Assessment**

The constraint-aware response is more closely aligned with the budget limitation because it begins with resource assessment and low-resource contingency planning. However, its visible text also introduces an additional assumption that there is no access to external power sources, which was not stated in the original scenario.

**Review finding:** The standard response risks relying on new infrastructure expenditure. The constraint-aware response better emphasizes resource-constrained alternatives, but it should avoid adding unsupported constraints.

### CMP01 — Routine Digital Records With Limited Shared Computers

**Explicit constraints**
- Several staff members share limited computers.
- No new computers can be purchased this quarter.

**Standard response**

The standard response recommends a computer rotation schedule, a centralized queue, staff training, and a computer check-in/check-out protocol.

**Constraint assessment**

These recommendations are compatible with the stated constraints because they improve allocation of existing equipment rather than requiring additional computers. The visible excerpt does not establish a clear violation.

**Constraint-aware response**

The constraint-aware response proposes shift-based scheduling and asks for details about staff numbers, record-entry frequency, user roles, and existing workflows.

**Assessment**

Both approaches appear compatible with the scenario in the visible text. The constraint-aware response provides a more explicit assessment of missing operational information, while the standard response offers several immediately actionable ideas.

**Review finding:** Based on the supplied excerpts, a constraint violation cannot be independently confirmed for `CMP01`. The recorded manual violation label should be checked against the complete model response and scoring notes before this example is presented as evidence.

### CMP02 — Scheduling Administrative Tasks With Few Workstations

**Explicit constraints**
- Only a few functioning workstations are available.
- Staff have competing administrative responsibilities.

**Standard response**

The standard response proposes using a shared digital calendar, digital task-management tools, and shift rotation.

**Constraint assessment**

A digital calendar or task-management platform may be useful, but the response does not establish whether these tools are already available, accessible during the relevant work periods, or feasible with the limited workstations. Requiring staff to use additional digital tools could create further competition for the same equipment.

Nevertheless, the visible excerpt does not prove that the suggested tools require new purchases or that staff cannot access them. The recommendation is therefore a potential feasibility concern, not a confirmed violation on the available evidence.

**Constraint-aware response**

The constraint-aware response explicitly lists the workstation shortage and competing responsibilities, then recommends prioritizing tasks by urgency and impact, assigning work based on staff availability, and scheduling tasks in blocks.

**Assessment**

The constraint-aware response is more directly focused on allocating scarce workstation time and balancing staff workload. Its suggestion of a shared digital task-management tool still depends on existing availability and access.

**Review finding:** The constraint-aware response gives more explicit attention to the central resource constraint. The claim that the standard response violated the constraint needs to be checked against the full response and the manual scoring record.

## 4. Quantitative Results

The manual evaluation covered 15 scenarios under each prompting condition, for 30 responses in total.

| Metric | Standard | Constraint-aware |
|---|---:|---:|
| Responses evaluated | 15 | 15 |
| Mean total score (maximum 8) | 6.53 | 6.87 |
| Mean constraint-adherence score (maximum 2) | 1.67 | 1.87 |
| Mean feasibility score (maximum 2) | 1.33 | 1.53 |
| Mean assumption-control score (maximum 2) | 1.33 | 1.47 |
| Mean adaptation-specificity score (maximum 2) | 1.93 | 2.00 |
| Recorded constraint violations | 4 | 0 |
| Recorded violation rate | 26.7% | 0.0% |

The mean total-score difference was **+0.33 points** in favor of constraint-aware prompting. Across the 15 paired scenarios, constraint-aware prompting scored higher in four, lower in one, and tied in ten.

These are descriptive results from a small, manually scored evaluation. They do not establish statistical significance, broad generalizability, or improved safety in real hospital operations.

## 5. Interpretation

The results suggest that explicitly instructing the model to identify operational constraints and missing information may help it produce responses that better reflect the stated scenario requirements.

Three observations are relevant:

1. **Explicit constraint identification:** The constraint-aware responses generally restate the operational constraints before proposing actions.
2. **Resource awareness:** Several constraint-aware responses foreground available resources and manual alternatives instead of immediately proposing infrastructure or additional tools.
3. **Remaining limitations:** Constraint-aware responses can still introduce unsupported assumptions, recommend tools whose availability is unknown, or produce generic plans that require further adaptation.

The qualitative review also shows why recorded violation labels should be supported by the full response and a clear explanation of the violated constraint. A recommendation that depends on an unknown resource is not automatically a proven violation unless the scenario or evaluation rubric rules that resource out.

## 6. Limitations

- Only 15 fictional scenarios were evaluated.
- The evaluation used one model identifier, `Qwen/Qwen3-4B`.
- The two prompting conditions were manually scored using a four-dimension rubric.
- The visible output excerpts are truncated, limiting independent verification of some violation labels.
- The results are descriptive and do not establish statistical significance.
- No real patient records or confidential hospital data were used in the scenarios provided.
- The study does not demonstrate clinical safety, real-world effectiveness, or generalization to other hospitals or models.

## 7. Reproducibility and Evidence Requirements

Before publication or submission, verify the following:

- [ ] Confirm the four recorded violation labels against the full model responses.
- [ ] Preserve the complete prompts and responses in the evaluation artifacts.
- [ ] Record the exact model revision if it can be established from the original execution environment.
- [ ] Document the actual generation settings used.
- [ ] Record relevant software and hardware environment details where available.
- [ ] Ensure that the summary metrics agree with the completed manual scoring file.
- [ ] Keep the evaluation scenarios fictional and free of identifiable patient information.

## 8. Conclusion

In this preliminary evaluation, constraint-aware prompting achieved a slightly higher mean manual score and had fewer recorded constraint violations than standard prompting. The findings support further investigation into structured prompting as a method for improving small language models' handling of operational constraints.

The results should be interpreted cautiously. The qualitative evidence must be checked against complete outputs and scoring notes before the four recorded violations are treated as confirmed examples. Further testing with more scenarios, independent evaluators, and clearly defined violation criteria would strengthen the conclusions.