# Prepix

A lightweight Python library for data preprocessing, quality checks, and ML utilities.

## Features

- ✅ missing_report()

## Example

```python
import prepix as pp
import pandas as pd

df = pd.read_csv("data.csv")

report = pp.missing_report(df)

print(report)

### Project Goal

Our goal is to develop a reusable Python library that **automates common EDA, data-preprocessing, and feature-engineering tasks** performed before training a Machine Learning model.

We want to reduce repetitive work for Data Science and ML developers and make the data-preparation process **faster, simpler, and more consistent**.

Our second goal is to make it a **continuously improving open-source project**. We will release an initial MVP, allow real users to use it in their ML projects, collect their feedback, and improve the library through future versions.

**In short:**

**Automate → Release → Get Real User Feedback → Improve → Repeat**
