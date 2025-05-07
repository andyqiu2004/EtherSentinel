# EtherSentinel - Blockchain Security Analysis System

EtherSentinel is a comprehensive blockchain security analysis system that leverages GNN-BERT for detecting and analyzing potential security threats in blockchain transactions and accounts.

## Features

1. **Transaction Analysis**
   - Pattern recognition in transaction flows
   - Anomaly detection
   - Smart contract interaction analysis

2. **Account Analysis**
   - Account behavior profiling
   - Risk scoring
   - Historical activity analysis

3. **Real-time Monitoring**
   - Live transaction monitoring
   - Instant threat detection
   - Alert system

## Project Structure

```
EtherSentinel/
├── models/                 # Trained models and model definitions
├── data/                  # Data storage and processing
├── src/                   # Source code
│   ├── analysis/         # Analysis modules
│   ├── models/           # Model architecture
│   ├── utils/            # Utility functions
│   └── monitoring/       # Real-time monitoring
├── config/               # Configuration files
├── notebooks/            # Jupyter notebooks for analysis
└── tests/               # Unit tests
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure your environment variables in `.env`
2. Run the analysis:
```bash
python src/main.py
```

## Model Training

The GNN-BERT model can be trained using:
```bash
python src/train.py
```

## License

MIT License 