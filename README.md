# University Admission Retrieval Chatbot System

## Overview

This project is a chatbot FAQ system designed to assist new students joining the university by providing relevant information regarding admissions. The chatbot leverages TF-IDF vectorization and cosine similarity to deliver accurate responses to user queries. It loads a dataset from a CSV file containing frequently asked questions and their corresponding answers. Using the `pandas`, `numpy`, and `scikit-learn` libraries, it vectorizes the questions and answers, and finds the most relevant answer to a user's question based on cosine similarity. This implementation demonstrates how machine learning techniques can be applied to create an interactive FAQ chatbot for university admissions.

## Setup

### Prerequisites

To set up the chatbot, ensure you have Python 3.x installed along with the required libraries (`pandas`, `numpy`, and `scikit-learn`). You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn
