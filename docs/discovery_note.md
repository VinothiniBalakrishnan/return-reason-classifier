# Dhaga & Co. Returns Reason Classifier Discovery Note Draft

**Date:** 2 October 2026 · **Status:** Draft for group review · **Project:** Returns Reason Classifier

**Problem and owner.** Neha, Category Head, cannot read all the return comments hidden under “Other,” so she lacks a counted view of which fit complaints deserve investigation. Today she reads a few hundred comments manually and believes most concern fit (client brief, printed p. 4).

**Evidence and cost of leaving it unsolved.** Returns are 31% overall; 44% of return reasons fall under “Other”; size charts differ by vendor (pp. 3–4). At 48,000 orders weekly (p. 2), the estimated workload is 48,000 × 31% × 44% ≈ **6,547 “Other” returns per week**, assuming the return rate is per order and both percentages use compatible populations. This indicates an investigation gap; manual-review hours and financial savings are unknown. The ₹120 figure concerns cash-on-delivery return-to-origin and must not be used as the cost of a customer return.

**Ranked shortlist.** The following assessment supports the selected project; it is a proposed rationale for group review, not a record of an earlier comparison. Ranking weighs evidence, usefulness, feasibility before the demo, and dependence on live systems.

| Rank | Problem and owner | Evidence and ranking rationale |
| --- | --- | --- |
| 1 | Unstructured return reasons — Neha | 44% “Other,” manual review bottleneck, variable size charts (pp. 3–4). Clear internal user and a bounded CSV workflow; useful language judgment without live integration. |
| 2 | Inconsistent catalogue attributes — Vivek | About 90 colour spellings and free-text fabric (p. 3). Strong normalization use case; approved vocabulary and catalogue review add scope, and its contribution to listing delays is unproven. |
| 3 | Repetitive order-status support — Arpita | 58% of tickets concern order status; first response averages nine hours (p. 4). Large opportunity, but accurate answers need current order data; ticket triage alone addresses only part of the problem. |
| 4 | Manual product-copy creation — Vivek | Around 200 descriptions weekly, inconsistent tone (p. 3). Feasible assistance, but publishing needs factual review and the brief does not isolate writing as the main listing bottleneck. |

**Chosen experiment.** Build an internal weekly CSV tool using LangChain to classify “Other” comments, extract fit direction and area where supported, and show complaint counts by vendor, category, and size with customer evidence. Flag ambiguous and failed rows for review. Rank counts as investigation priorities, not return rates or proven fixes. A keyword-only baseline is cheaper and should be compared on the same labelled examples; mixed-language and multi-reason comments are the proposed justification for model use.

**Success and measurement.** Proposed primary target: at least **90% top-level accuracy** on a reserved, manually labelled test set, including Unclear examples. Score every test row; failed or unclassified predictions count as incorrect. Report specific-label coverage, fit-direction accuracy, failures, runtime, and measured cost separately. The usability check is whether a teammate acting as Neha can run the file and identify a supported investigation without coaching. Existing return comments support evaluation; future impact measurement needs returns linked to sales or deliveries and recorded interventions. Reduced returns are not an MVP claim.

**Biggest assumption and next evidence.** Comments contain enough specific, recurring information to support useful group-level investigation. Audit the pending CSV for usable text, metadata, and group sizes. Predominantly blank or vague comments, missing vendor/size fields, or groups too sparse to distinguish would undermine the proposed action list. Dataset provenance and schema remain unverified. Rehearsal: **3 October, 9:30 AM IST**; demo: **5:30 PM IST**. Group agreement is pending.

**Source:** FDE Academy, *Mini Project 1 — The Dhaga & Co. Engagement*, printed pp. 2–5. Business figures above are case-study facts; calculations, rankings, and targets are explicitly proposed or conditional.

