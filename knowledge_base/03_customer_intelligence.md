# Customer Intelligence

## Response Probability

The XGBoost model estimates the likelihood of a positive customer response.

Higher probability means stronger predicted response opportunity relative to other customers.

The probability is not a guarantee.

## Financial Value Proxy

The project uses customer `balance` as a Financial Value Proxy.

Balance is NOT:
- Customer Lifetime Value
- revenue
- profit
- expected future value

Always call it the Financial Value Proxy.

## Priority Score

Priority Score combines:
- 50% response score
- 30% value score
- 20% segment priority score

Conceptually:

Priority Score =
0.50 × Response Score
+ 0.30 × Value Score
+ 0.20 × Segment Score

Response score is based on response probability.

Value score is based on financial-value percentile.

Segment score comes from business-defined segment priority weights.

## Contact Fatigue

Contact fatigue is a rule-based indicator using contact history.

It is not a psychological or medical measurement.

More contact does not necessarily produce more conversion.

## Next Best Action

Platform rules:

1. High fatigue + response probability below 0.60 → Pause contact.
2. Response probability at least 0.60 → Priority sales follow-up.
3. Value percentile at least 0.75 + response probability at least 0.40 → Personalized high-value offer.
4. Response probability at least 0.30 → Nurture with improved messaging.
5. Otherwise → Low-cost digital nurture.

These are platform business rules. The LLM should explain them rather than silently replacing them.
