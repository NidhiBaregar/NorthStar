# Project Context

This project is an AI-powered customer intelligence and marketing decision-support platform using the UCI Bank Marketing dataset.

The platform combines:
- data processing
- XGBoost response prediction
- K-Means customer segmentation
- business rules
- campaign analysis
- RAG
- LangChain
- Grok LLM

The system helps a marketing manager answer:
1. Who should we target?
2. Which customers deserve sales attention?
3. How should each customer segment be approached?
4. What is happening in campaign performance?
5. Why was a recommendation made?
6. What should the manager do next?

## Architecture

Customer Data
→ ML Models
→ Customer Intelligence
→ Business Rules
→ Streamlit Dashboard
→ RAG retrieves relevant knowledge
→ Grok explains and recommends.

## Layer Responsibilities

Python/ML produces factual customer, prediction and segmentation outputs.

Business logic converts those outputs into operational recommendations.

RAG retrieves approved marketing theory and project knowledge.

Grok explains the evidence, applies the appropriate framework and generates strategic reasoning.

The LLM must not replace the underlying ML model or business rules.
