Lendora is an intelligent credit scoring platform that combines machine learning, behavioral analysis, OCR, and generative AI to assess an individual's creditworthiness. It is designed for lenders, fintechs, and microfinance institutions to streamline credit evaluation and minimize risk in real time.

🔍 Overview
Lendora integrates predictive analytics with modern AI capabilities to:

Predict loan default probability

Extract applicant data using OCR from uploaded documents

Generate credit summaries using Anthropic's Claude API

Enforce secure access using Role-Based Access Control (RBAC)

Provide a user-friendly, role-sensitive dashboard experience

🧠 Features
Credit Risk Prediction: ML-powered predictions based on behavioral and demographic data

Document OCR: Auto-extracts and parses financial or ID documents using Tesseract OCR

Generative AI Summaries: Converts data into natural language insights using Claude (Anthropic API)

RBAC (Role-Based Access Control): Secure role-layered access for admin, loan officers, and viewers

Interactive Dashboards: Monitor loan applications, approval rates, and model performance

🧱 Tech Stack
FastAPI – Backend RESTful API

Streamlit – Frontend dashboard and input UI

Anthropic Claude API – For generative summaries

Tesseract OCR – For document image parsing

PostgreSQL / Google Sheets – For database + logs

Scikit-learn  – For ML modeling

Looker Studio  – For external BI dashboard integration

