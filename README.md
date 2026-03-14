# Blinkit Sales Intelligence

End-to-end Python analytics pipeline for analyzing Blinkit grocery retail datasets using statistical analysis, clustering, anomaly detection, forecasting, and automated PDF reporting.

This project processes retail data, extracts insights, builds machine learning models, generates visual analytics, and automatically compiles a professional report summarizing operational and sales patterns.

## Features

* Automated data preprocessing and cleaning
* Time-based analytics and trend analysis
* Correlation and statistical insight generation
* KMeans clustering for behavioral grouping
* Isolation Forest anomaly detection
* Random Forest feature importance analysis
* Time-series forecasting using Exponential Smoothing
* Automated visualization generation
* Automatic PDF report generation with charts and insights

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Statsmodels, ReportLab

## Project Structure

```
blinkit-sales-intelligence
├── Analyzer.py
├── BlinkIT Grocery Data set 1.xlsx
├── Output
│   ├── charts
│   ├── models
│   └── Retail_Sales_Report.pdf
```

## Installation

```
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels reportlab
```

## Usage

Run the analyzer script from the project directory:

```
python Analyzer.py
```

The script automatically detects the dataset, performs the analysis, generates charts, and creates a final PDF report inside the Output folder.

## Output

The pipeline produces:

* Multiple analytical charts
* Cluster analysis results
* Feature importance insights
* Time series forecasts
* A full automated analytics report in PDF format

## Purpose

This repository demonstrates a practical retail analytics workflow combining data engineering, machine learning, and automated reporting within a single Python pipeline.
