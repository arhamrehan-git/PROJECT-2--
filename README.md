# PROJECT-2--
Machine learning web app that predicts loan default risk, explains each decision, suggests counter-offers, and shows the profit impact of lending policies. Built with Python, scikit-learn and Streamlit.
# Loan Risk AI

**Loan Default Prediction and Decision Support System**

[![Open the live app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)

Loan Risk AI is a machine learning web app that estimates the probability that a loan applicant will default. It turns that probability into an approve or reject decision, explains the decision, suggests a smaller loan amount when a loan is rejected, and shows what each lending policy is worth in money.

Project report for B.Tech CSE (Artificial Intelligence), Jamia Hamdard, New Delhi.
Author: Arham Rehan (Enrollment No. 2023-350-005) | Supervisor: Prof. Harleen Kaur

## Contents

- [Features](#features)
- [Try it in 1 minute](#try-it-in-1-minute)
- [Results](#results)
- [How it works](#how-it-works)
- [Run it locally](#run-it-locally)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Limitations](#limitations)

## Features

- **Risk prediction:** probability of default from 5 inputs: annual income, loan amount, interest rate, home ownership and prior default on file.
- **Decision rule:** approve or reject using either a risk cut-off slider or a profit based break-even rule.
- **Explanation:** a chart showing how each input raises or lowers the risk compared with a typical applicant.
- **Counter-offer:** for a rejected loan, the largest smaller loan that would be approved (or a clear message that no loan size works). For an approved loan, how much more could be borrowed.
- **Business Impact:** simulated total profit for different cut-offs on held-out loans, using profit and loss percentages set in the sidebar.
- **Model and Pipeline:** ROC-AUC, recall, false positive rate, precision, ROC curve, confusion matrix and feature importance.

## Try it in 1 minute

Open the live app, then use **Demo profiles** in the sidebar:

| Profile | Income | Loan | Rate | Home | Prior default | What you will see |
| --- | --- | --- | --- | --- | --- | --- |
| Safe profile | 90,000 | 6,000 | 7.5% | Mortgage | No | Approved, risk about 1.4% |
| Loan too large | 40,000 | 15,000 | 13% | Rent | No | Rejected, counter-offer of about 12,000 |
| Risky profile | 22,000 | 15,000 | 18.5% | Rent | Yes | Rejected, no loan size would be approved |

Then switch the sidebar decision rule to **Maximise expected profit** and open the **Business Impact** tab.

## Results

Measured on a held-out test set of about 6,500 loans that the model never saw:

| Measure | Value |
| --- | --- |
| ROC-AUC | 0.917 |
| Bad loans caught (recall), cut-off 25% | about 79% |
| Good loans wrongly flagged, cut-off 25% | about 11.5% |
| Precision, cut-off 25% | about 66% |

The cut-off can be changed in the app, and these measures update with it.

## How it works

1. The dataset (about 32,500 loans, about 22% defaulted) is split 60/20/20 into training, validation and test sets, keeping the same default rate in each part.
2. A scikit-learn pipeline one-hot encodes the two categorical fields and passes the numeric fields to a `HistGradientBoostingClassifier` (gradient boosting with early stopping).
3. The default cut-off is chosen on the validation set using Youden's J (true positive rate minus false positive rate).
4. Final scores are measured on the test set.
5. Expected profit of a loan is `EP = L * ((1 - p) * g - p * l)`, where L is the loan amount, p the probability of default, g the profit rate if repaid and l the loss rate if defaulted. The break-even risk is `g / (g + l)`.

Age is deliberately not used, because it is a protected attribute in lending decisions.

## Run it locally

Requires Python 3.9 or higher.

```
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. On the first run the app trains the model by itself (under a minute). You can also train it manually with `python train.py`.

## Project structure

```
loan-risk-ai/
  app.py                    Streamlit web interface
  train.py                  data loading, model, training and explanation helpers
  credit_risk_dataset.csv   historical loan data
  requirements.txt          Python packages
  README.md
```

The `artifacts/` folder (the saved model and report) is created automatically and is not stored in the repository.

## Dataset

Credit Risk Dataset on Kaggle: https://www.kaggle.com/datasets/laotse/credit-risk-dataset. Check the dataset page for its license before redistributing it.

## Limitations

- The model learns from historical data, so it finds patterns and not causes, and past bias can carry over.
- Only 5 inputs are used. The largest loan in the training data is 35,000, so scores for much larger loans are less reliable.
- The interest rate is an input, but in real lending it is often set after the applicant has been assessed.
- Profit and loss percentages are assumptions entered by the user, not real bank figures.
- The app supports a credit officer and does not replace one. It is an educational project, not financial advice.
