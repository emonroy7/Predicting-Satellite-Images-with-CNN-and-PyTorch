# Predicting-Satellite-Images-with-CNN-and-PyTorch

# 📈 Forecasting Stock Market Trends

A machine learning–based system for predicting stock market trends using multiple predictive models, applied across major tech stocks. Includes a Graphical User Interface (GUI) for running four types of financial analysis.

---

## 🧠 Models Used

| Analysis Type | Model |
|---|---|
| Fundamental Analysis | Feedforward Neural Network |
| Trading Strategy | Random Forest Classifier |
| Risk Analysis | Long Short-Term Memory (LSTM) |
| Portfolio Analysis | Random Forest Regressor |

---

## 📊 Stocks Analyzed

| Ticker | Company |
|---|---|
| `AAPL` | Apple Inc. |
| `MSFT` | Microsoft Corporation |
| `AMD` | Advanced Micro Devices, Inc. |
| `INTC` | Intel Corporation |
| `NVDA` | NVIDIA Corporation |

---

## 🔧 Tech Stack

- **Language:** Python
- **Data Source:** [`yfinance`](https://pypi.org/project/yfinance/) — historical stock price data
- **ML Libraries:** TensorFlow / Keras, scikit-learn
- **GUI:** Tkinter (or specify your framework)
- **Other:** NumPy, Pandas, Matplotlib

---

## 🔄 Workflow

```
Data Collection → Data Preprocessing → Model Selection → Model Training → Model Evaluation → Prediction & Analysis
```

1. **Data Collection** — Pull historical stock data via `yfinance`
2. **Data Preprocessing** — Clean, normalize, and prepare time-series data
3. **Model Selection** — Choose model based on analysis type (see table above)
4. **Model Training** — Fit models on historical data and tune hyperparameters
5. **Model Evaluation** — Assess accuracy using relevant metrics (MSE, accuracy, etc.)
6. **Prediction & Analysis** — Generate forecasts and visualize trend outputs

---

## 🖥️ GUI Features

The application provides a GUI with four analysis modules:

- **Fundamental Analysis** — Evaluates intrinsic stock value using financial indicators
- **Trading Strategy** — Applies technical analysis for short-term price movement predictions
- **Risk Analysis** — Identifies and assesses investment risks using LSTM
- **Portfolio Analysis** — Optimizes a collection of assets to meet financial goals

Each module has a dedicated **Run Analysis** button for on-demand predictions.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install yfinance numpy pandas matplotlib scikit-learn tensorflow
```

### Run the Application

```bash
python main.py
```

---

## ⚖️ Ethics & Responsible AI

This project acknowledges the ethical responsibilities involved in AI-driven financial modeling:

- **Transparency** — Models and their limitations are clearly documented
- **Fairness** — Potential biases in training data are considered
- **Accountability** — Human judgment is encouraged alongside algorithmic outputs
- **No financial advice** — Predictions are for educational and research purposes only; this is not investment advice

---

## 🔭 Future Directions

- Explore advanced deep learning architectures (e.g., Transformer-based models)
- Integrate alternative data sources (news sentiment, social media signals)
- Implement Explainable AI (XAI) for interpretable predictions
- Incorporate behavioral finance and market psychology indicators

---

## 👤 Author

**Emon Roy**

---

## 📄 License

This project is for educational purposes. See `LICENSE` for details.
