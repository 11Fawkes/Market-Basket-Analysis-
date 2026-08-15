# Market Basket Analysis

**Which products get bought together — and what should retailers do about it?**

Association rule mining on retail transaction data to surface high-confidence product pairs for bundling, cross-sell, and store layout decisions.

## The Problem

Retailers sit on transaction logs but rarely analyse which products are bought together. Without this, bundling strategies are guesswork and shelf placement is based on convention, not data.

## What This Does

Loads transactional purchase data → applies the Apriori algorithm → surfaces product association rules ranked by support, confidence, and lift. Three analysis views:

- **First Choices** — bar chart of most frequently purchased items
- **Second Choices** — network graph showing top 15 item relationships
- **Apriori Rules** — full association rules table with support, confidence, and lift

## Key Features

- Apriori algorithm via mlxtend with configurable support threshold (default 0.5%)
- Association rules filtered by lift ≥ 1.2 (stronger than random co-occurrence)
- NetworkX spring-layout graph for intuitive product relationship visualisation
- Built as a Streamlit app for interactive exploration

## Tech Stack

Python · Streamlit · mlxtend (Apriori) · Pandas · NetworkX · Matplotlib

## Run Locally

```bash
pip install streamlit mlxtend pandas networkx matplotlib numpy
streamlit run DA3.py
```

> Note: requires a CSV file with transactional data. Place your data file in the `data/` directory.

## About

Built by Dhruv Kumar — Business Analyst, Berlin.
