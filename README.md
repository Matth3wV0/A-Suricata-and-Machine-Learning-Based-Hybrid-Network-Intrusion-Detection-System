# Hybrid Network Intrusion Detection System (NIDS)

A sophisticated hybrid Network Intrusion Detection System that combines signature-based detection (Suricata) with machine learning-based anomaly detection for comprehensive network security monitoring.

## Features

- **Hybrid Detection Approach**
  - Signature-based detection using Suricata
  - Machine learning-based anomaly detection
  - Behavioral analysis
  - Session-aware detection

- **Multiple ML Models**
  - Decision Tree
  - Random Forest
  - XGBoost
  - Ensemble approach for improved accuracy

- **Advanced Features**
  - Real-time monitoring
  - Historical analysis
  - Automated alerting (including Telegram integration)
  - Detailed logging
  - Session tracking
  - Behavioral analysis
  - Flow analysis

- **Alert System**
  - Multi-channel alerting
  - Severity-based notifications
  - Detailed threat information
  - Telegram integration

## Prerequisites

- Python 3.8 or higher
- Suricata
- Required Python packages (see requirements.txt)
- Telegram Bot Token (for alerting)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Matth3wV0/A-Suricata-and-Machine-Learning-Based-Hybrid-Network-Intrusion-Detection-System
   cd A-Suricata-and-Machine-Learning-Based-Hybrid-Network-Intrusion-Detection-System
   ```

2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with Telegram credentials (if you want to use Telegram alerts):
   ```
   TELEGRAM_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   API_ID=your_telegram_api_id
   API_HASH=your_telegram_api_hash
   ```

4. Create a model directory (default: `./model`):
   ```
   mkdir -p ./model
   ```


## Usage

### Training the Models

```bash
python hybrid_nids.py --train path/to/dataset.csv --model_dir ./model
```

### Real-time Monitoring

```bash
python hybrid_nids.py --realtime path/to/suricata.json --telegram
```

### Analyzing Suricata Logs

```bash
python hybrid_nids.py --analyze path/to/suricata.json --output results.csv
```

### Command Line Arguments

- `--train`: Train models using specified dataset
- `--analyze`: Analyze Suricata logs
- `--realtime`: Enable real-time monitoring
- `--model_dir`: Specify model directory
- `--output`: Specify output file
- `--telegram`: Enable Telegram alerts

## Project Structure

```
.
├── hybrid_nids.py          # Main system orchestrator
├── training.ipynb          # Interactive training
├── utils/                  # Utility modules
│   ├── suricata_parser.py
│   ├── suricata_flows.py
│   ├── telegram_alert.py
│   ├── adaptive_flow_features.py # Flow feature extraction
│   ├── session_manager.py      # Session management
│   ├── behavioral_analyzer.py  # Behavioral analysis
│   ├── anomaly_detector.py     # Anomaly detection
│   ├── flow_finalizer.py       # Flow processing
│   ├── service_whitelist.py    # Service whitelisting
│   ├── dataset_balancer.py     # Dataset balancing
│   └── suricata_debug.py       # Suricata debugging
├── requirements.txt        # Dependencies
└── .env                    # Configuration
```

## Key Components

1. **Signature-based Detection**
   - Uses Suricata for known attack detection
   - Processes network traffic
   - Generates alerts based on rules

2. **Machine Learning Models**
   - Multiple model approach
   - Feature extraction and processing
   - Anomaly detection
   - Model training and evaluation

3. **Session Management**
   - Tracks network sessions
   - Maintains connection states
   - Correlates related events

4. **Behavioral Analysis**
   - Pattern recognition
   - Anomaly detection
   - Baseline establishment
   - Deviation analysis

5. **Flow Analysis**
   - Network flow processing
   - Feature extraction
   - Flow finalization
   - Analysis integration

## Alert System

The system provides comprehensive alerting capabilities:
- Real-time alerts
- Severity levels
- Detailed threat information
- Multiple notification channels
- Telegram integration

## Logging

System logs are maintained in:
- Console output
- `hybrid_nids.log` file
- Alert logs
- Flow analysis results

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Acknowledgments

- Suricata project
- Python community
- Machine learning libraries
- Open-source contributors
- CICIDS2017 dataset creators for providing training data
- Suricata team for their excellent IDS engine
- The authors of research paper "A Suricata and Machine Learning Based Hybrid Network Intrusion Detection System"

## Contact

Created with ❤️ by M4tth3wV0