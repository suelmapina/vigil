# 🛡️ Vigil: Cost-Aware Fraud Screening for Account Opening

> **Work in progress.** Done so far: data, drift analysis, model comparison, cost-based decisions,
> reason codes and fairness. Next: monitoring, API and demo.

## The business problem

Every online account application gets a decision in seconds: approve, send to an analyst, or
decline. Missing a fraudster costs the credit line that is opened. Flagging a legitimate customer
costs analyst time and, often, the customer. And the fraud mix keeps moving, so a model and a
threshold tuned today can quietly stop working next month.

## Data

[Bank Account Fraud (BAF)](https://github.com/feedzai/bank-account-fraud), base variant: 1,000,000
applications over 8 months, 30 features, 1.10 % fraud. It was generated from real bank data with
privacy protection, and published by Feedzai at NeurIPS 2022. Licensed CC BY-NC-SA 4.0; downloaded
from Kaggle with `make data` and never committed.

**Split by time, as in production.** Fit on months 0 to 4, tune on month 5 (early stopping,
calibration, thresholds), report on months 6 and 7, which no model or decision has seen.

**Cleaning.** Some columns use −1 for "missing". Missingness is a signal (no previous address on
file: 1.42 % fraud vs 0.31 %), so these become explicit missing values rather than a small number.
One column is constant and is dropped. **Age is kept to measure fairness but is never used to
score** (section 4 explains why).

## 1. The data moves

| Month | 0 | 1 | 2 | 3 | 4 | 5 | **6** | **7** |
|---|---|---|---|---|---|---|---|---|
| Applications | 132k | 128k | 137k | 151k | 128k | 119k | 108k | 97k |
| Fraud rate | 1.13 % | 0.94 % | 0.87 % | 0.92 % | 1.14 % | 1.18 % | **1.34 %** | **1.47 %** |

- **Fraud is 41 % more frequent in the test months** (1.40 % vs 0.998 % in the fit months).
- **Three features shift a lot.** The application velocity counts (6 h, 24 h, 4 weeks) have a
  population stability index of 1.4 to 4.0 in the test months; anything above 0.25 is
  usually treated as a large shift. They had already moved by month 5, so that is visible before
  any test data exists.

## 2. Models

Four approaches, each scored on the two unseen months (205k applications, 2,878 frauds). The
headline metric is the one used in the BAF paper: share of fraud caught when 5 % of legitimate
applicants are flagged.

| Model | Fraud caught at 5 % FPR | PR-AUC | ROC-AUC | Fit time |
|---|---|---|---|---|
| Rules engine (7 red flags) | 21.4 % | 0.04 | 0.74 | |
| Logistic regression | 50.1 % | 0.16 | 0.88 | 12 s |
| LightGBM, all features | 54.3 % | 0.19 | 0.89 | 47 s |
| **LightGBM, stable features** | **54.4 %** | **0.19** | **0.89** | 62 s |

95 % intervals from 200 paired bootstrap resamples: LightGBM catches **+4.4 pp** more fraud than
logistic regression (+2.9 to +5.8) and more than twice as much as the rules engine.

**Why ship the model without the drifting features.** Both LightGBM versions rank applications
equally well. The difference shows up when the threshold is set on month 5 and left alone, as it
would be in production:

| Threshold fixed on month 5 | FPR, month 6 | FPR, month 7 | FPR, both | Fraud caught |
|---|---|---|---|---|
| Target | 5.0 % | 5.0 % | 5.0 % | |
| LightGBM, all features | 6.6 % | 5.6 % | 6.1 % | 58.7 % |
| **LightGBM, stable features** | **6.1 %** | **4.7 %** | **5.4 %** | 56.5 % |

With all features, 23 % more legitimate customers are flagged than planned. Dropping the three
velocity features keeps the error closer to budget at no cost in ranking quality. It does not
remove the shift, which is why monitoring and threshold review are part of the design.

**Calibration.** Raw scores predict 0.99 % fraud in the test months; the true rate is 1.40 %.
Isotonic calibration on month 5 brings this to 1.27 %. It cannot fully catch up, because fraud
keeps rising after month 5.

## 3. From scores to decisions, in euros

A score is not a decision. Each application gets the cheapest of three actions, given its fraud
probability *p* and its proposed credit limit *L*:

| Action | Expected cost |
|---|---|
| Approve | *p* × *L* (a fraudster draws the credit line) |
| Review | €20 analyst time + 10 % of the fraud loss that slips past the analyst + €2 delay for a good customer |
| Decline | (1 − *p*) × €100 value of a good customer turned away |

Analysts are a fixed resource: they can review **2.5 %** of applications. When more applications
are worth reviewing, the ones where a review saves the most go first; the rest are approved or
declined, whichever is cheaper. All costs are illustrative defaults and are adjustable.

**Result on the two unseen months** (205,011 applications, 2,878 frauds, same analyst capacity for
every policy):

| Policy | Total cost | Fraud stopped | Good customers declined | Extra cost vs Vigil (95 % CI) |
|---|---|---|---|---|
| Approve everyone | €2.32M | 0 % | 0 | +€1.13M (1.04M to 1.22M) |
| Rules: analysts review the top alerts | €2.15M | 12 % | 0 | +€963k (887k to 1.05M) |
| Model: analysts review the top scores | €1.42M | 33 % | 0 | +€229k (193k to 269k) |
| Model: decline everything above 5 % FPR | €1.83M | 57 % | 10,940 | +€638k (612k to 663k) |
| **Vigil: cost-based decision** | **€1.19M** | 34 % | 896 | |

95 % intervals from 200 paired bootstrap resamples.

- **Against the rules setup, Vigil cuts the cost of fraud and screening by 45 %** (€963k over two
  months) with the same analyst team.
- **Most of the gain comes from the model; the last €229k from the decision rule.** Same model,
  same analysts: using expected cost instead of the raw score ranking sends analysts to the
  applications where a mistake would be most expensive, and declines the clearest cases outright.
- **The 5 % FPR rule stops the most fraud and is still €638k more expensive**, because it turns away
  10,940 legitimate applicants. A benchmark operating point is not a business one.
- **Exposure drives the decision.** Applications asking for a €1,500 to €2,100 credit line are
  reviewed 16 % of the time and declined 5 %; those asking for €200 are approved 99.9 % of the
  time. A €2,000 line is worth checking at a fraud risk that a €200 line is not.
- **Vigil is the cheapest policy in all 25 scenarios tested** (value of a good customer from €25 to
  €400, analyst capacity from 0.5 % to 10 %).
- **Calibration barely changes the result here** (+€10k without it, −2k to +21k), because the
  decisions depend mostly on the ranking and the credit limit.

## 4. Reasons and fairness

**Every review comes with its reasons.** The three features that pushed an application towards
fraud, in plain words, from exact TreeSHAP values computed by LightGBM itself. A real example from
the test months:

> Fraud probability 13 %, credit limit €500 → **review** (it was fraud)
> 1. Housing status: BA
> 2. Name and email similarity: 0.20
> 3. Request from abroad: 1

Categories are anonymised in the dataset (BA, CA and so on); in a bank they would be readable.

**Who carries the false alarms.** Fraud is 2.7 times more common among applicants aged 50 or more
(2.9 % vs 1.1 %). A model that gets the risk right for both groups will therefore stop more of
their legitimate applicants too: equal error rates and equal calibration cannot both hold when base
rates differ. Any fix is a choice, so each option is priced:

| Option | Good applicants stopped, under 50 / 50+ | Balance (1 = equal) | Fraud stopped | Cost vs Vigil (95 % CI) |
|---|---|---|---|---|
| Age used as a model input | 1.9 % / 7.2 % | 0.26 | 34.4 % | −€7k (−29k to +12k) |
| **Vigil: age never used to score** | 2.0 % / 6.1 % | 0.34 | 34.1 % | |
| Equal burden: lower the risk of the 50+ group | 2.3 % / 3.1 % | 0.74 | 31.0 % | +€63k (43k to 86k) |

"Stopped" means reviewed or declined.

- **Dropping age costs nothing and reduces the gap**, so Vigil ships without it. At the paper's
  5 % FPR operating point the ratio improves from 0.30 to 0.40. It does not close the gap, because
  other features carry part of the age signal.
- **Closing most of the gap costs about €63k over two months (+5 %)** and stops less fraud among
  older applicants (47 % to 34 %). It also means using age *in the decision*, which is
  differential treatment. Whether that trade is acceptable is a decision for legal and compliance,
  not for the model; the analysis gives them the price.
- **Fairness also drifts.** The equal burden adjustment was set to balance month 5 exactly; on the
  test months the balance is 0.74, not 1. Group error rates need monitoring like any other metric.

## Quickstart

```bash
make install       # uv sync, Python 3.11
make data          # BAF base from Kaggle (needs a Kaggle API token in ~/.kaggle)
make all           # drift, models, decisions, reasons and fairness (~7 min)
make test          # unit tests
make mlflow        # experiment tracking UI
```

## Citation

> Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R. P., Gama, J., and Bizarro, P.
> (2022). Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation.
> NeurIPS 2022.
