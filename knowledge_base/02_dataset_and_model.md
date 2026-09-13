# Dataset and Model Knowledge

## Dataset

The project uses the UCI Bank Marketing dataset.

Important fields include:
- age
- job
- marital
- education
- default
- balance
- housing
- loan
- contact
- day
- month
- campaign
- pdays
- previous
- poutcome
- y

The target variable `y` indicates whether the customer subscribed to the offered product.

## Response Prediction

The final response model is a balanced XGBoost classifier.

Response probability is an estimated likelihood of a positive response.

It is NOT:
- a guaranteed conversion
- guaranteed revenue
- a causal estimate of campaign impact

Use it together with value, segment, fatigue and campaign context.

## Segmentation

K-Means creates four business-oriented customer segments.

Four clusters were selected primarily for business interpretability rather than claiming that four is statistically optimal.

## Model Guardrails

Do not invent:
- model metrics
- feature effects
- customer values
- probabilities
- causal relationships

unless they are supplied by the application.
