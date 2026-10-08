# Freight Rate Prediction

Predicts the posted rate of a freight load from its lane, distance, equipment, weight, date and two market signals. Built for the Spotter Machine Learning Engineer assessment.

The model predicts log(rate) with an even blend of a linear regression and gradient boosting (scikit-learn). It is validated with rolling time-based folds, and on those folds the median error is roughly 1.5-1.8%. The full write-up is in `outputs/report.pdf`.

## Project layout

```
data/                              input CSVs (not committed, see below)
src/features.py                    cleaning, feature engineering, outlier flagging
src/baseline.py                    simple baselines and a two-fold check
src/train_final.py                 tuning, three-fold validation, final predictions
src/december_chart.py              predictions for the fixed December chart inputs
outputs/                           tuning results and the report
score.py                           scorer provided with the assessment
validation_predictions.csv         final predictions for validation.csv
```

## Setup

Python 3.10+ is recommended.

```
python -m venv .venv
.venv\Scripts\activate          # on Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Put these files in `data/` (the CSVs are not committed because they belong to the assessment):

- `train_test.csv`
- `validation.csv`
- `validation_predictions_template.csv`
- `december_chart_inputs.csv`

## Run

Run everything from the project root.

```
python src/baseline.py          # optional: baselines and a quick two-fold comparison
python src/train_final.py       # tunes, validates, writes validation_predictions.csv
python src/december_chart.py    # fills predicted_rate in data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`train_final.py` takes a minute or two. It writes `validation_predictions.csv` and `outputs/tuning_results.csv`. The scorer checks both files and creates `scorer_results/candidate_december.png`.

## Approach in short

- **Cleaning:** negative weights are treated as sign errors, missing weights get the training median, and missing market_index values get that day's median. Missing flags are kept as features.
- **Noisy rates:** about 1.4% of training loads have rates far from anything the features explain. They are left out when fitting but kept when scoring.
- **Validation:** each fold trains on earlier months and validates on the next two (train Jan-Apr / validate May-Jun, and so on), since the real task is predicting the two months after the training data.
- **Features:** log distance, short-haul flags, a flatbed premium that builds through the last month of each quarter, the gap of quote_signal from its middle value, and day-level market context.
- **December chart:** market_index and quote_signal for each December day come from the daily median of the validation loads.