# 🛡️ Hybrid Network Intrusion Detection System (NIDS)

## Hệ Thống Phát Hiện Xâm Nhập Mạng Kết Hợp Suricata và Machine Learning

Một hệ thống phát hiện xâm nhập mạng hybrid tiên tiến, kết hợp **phát hiện dựa trên chữ ký (Suricata)** với **phát hiện bất thường dựa trên Machine Learning** để giám sát an ninh mạng toàn diện.

---

## 📋 Mục Lục

- [Tổng Quan Kiến Trúc](#tổng-quan-kiến-trúc)
- [Luồng Xử Lý Hệ Thống](#luồng-xử-lý-hệ-thống)
- [Chi Tiết Các Thành Phần Python & ML](#chi-tiết-các-thành-phần-python--ml)
  - [1. Data Pipeline & Preprocessing](#1-data-pipeline--preprocessing)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Machine Learning Models](#3-machine-learning-models)
  - [4. Session Management](#4-session-management)
  - [5. Behavioral Analysis](#5-behavioral-analysis)
  - [6. Anomaly Detection Engine](#6-anomaly-detection-engine)
  - [7. Flow Finalizer](#7-flow-finalizer)
- [Dataset & Training](#dataset--training)
- [Cài Đặt & Sử Dụng](#cài-đặt--sử-dụng)
- [Cấu Trúc Project](#cấu-trúc-project)

---

## 🏗️ Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID NIDS ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐     ┌──────────────────┐     ┌─────────────────────────────┐ │
│  │   Suricata   │────▶│  Suricata Parser │────▶│    Session Manager          │ │
│  │  eve.json    │     │  (JSON → Object) │     │  (Aggregate Multi-Events)   │ │
│  └──────────────┘     └──────────────────┘     └─────────────┬───────────────┘ │
│                                                               │                 │
│                                                               ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        FEATURE EXTRACTION LAYER                             ││
│  │  ┌─────────────────────────┐    ┌───────────────────────────────────────┐  ││
│  │  │ AdaptiveFlowFeatures    │    │ Application Layer Features            │  ││
│  │  │ • dest_port             │    │ • HTTP (methods, status, hosts, URIs) │  ││
│  │  │ • duration              │    │ • DNS (queries, answers, failures)    │  ││
│  │  │ • total_fwd/bwd_packets │    │ • TLS (SNI, versions, certificates)   │  ││
│  │  │ • total_fwd/bwd_bytes   │    │ • SSH (client/server versions)        │  ││
│  │  │ • flow_bytes_per_sec    │    │ • File transfers                      │  ││
│  │  │ • flow_packets_per_sec  │    └───────────────────────────────────────┘  ││
│  │  │ • down_up_ratio         │                                                ││
│  │  └─────────────────────────┘                                                ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                   │                                             │
│                                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                         DETECTION LAYER                                     ││
│  │  ┌─────────────────────────┐    ┌───────────────────────────────────────┐  ││
│  │  │   ML ANOMALY DETECTOR   │    │    STATISTICAL ANOMALY DETECTOR       │  ││
│  │  │ ┌─────────────────────┐ │    │ • Z-Score Analysis (threshold: 4.0)   │  ││
│  │  │ │ Decision Tree (20%) │ │    │ • IQR Outlier Detection               │  ││
│  │  │ │ Random Forest (30%) │ │    │ • Feature-weighted Scoring            │  ││
│  │  │ │ XGBoost (50%)       │ │    │ • Baseline Comparison                 │  ││
│  │  │ └─────────────────────┘ │    └───────────────────────────────────────┘  ││
│  │  │   Voting: ≥2/3 models   │                                                ││
│  │  └─────────────────────────┘                                                ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                   │                                             │
│                                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                      BEHAVIORAL ANALYSIS LAYER                              ││
│  │  ┌─────────────────────────┐    ┌───────────────────────────────────────┐  ││
│  │  │   IP BEHAVIOR TRACKING  │    │     ATTACK PATTERN DETECTION          │  ││
│  │  │ • Flow rates            │    │ • Port Scan Detection                 │  ││
│  │  │ • Destination diversity │    │ • Host Scan Detection                 │  ││
│  │  │ • Volume metrics        │    │ • Brute Force Detection               │  ││
│  │  │ • Protocol distribution │    │ • Zero-byte Pattern Detection         │  ││
│  │  │ • Auth failure tracking │    │ • Volume Anomaly Detection            │  ││
│  │  └─────────────────────────┘    └───────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                   │                                             │
│                                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                          ALERT SYSTEM                                       ││
│  │  • Console Logging  • File Logging  • Telegram Notifications  • CSV Export ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Luồng Xử Lý Hệ Thống

### Flow Chi Tiết Từ Input Đến Output

```
                          SYSTEM FLOW DIAGRAM
                          
    ┌─────────────────────────────────────────────────────────────┐
    │                    1. DATA INGESTION                        │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Suricata eve.json (Real-time)                              │
    │       │                                                     │
    │       ▼                                                     │
    │  SuricataParser.process_line()                              │
    │       │ ─── Parse JSON → Dataclass Objects                  │
    │       │     (SuricataFlow, SuricataHTTP, SuricataDNS,       │
    │       │      SuricataTLS, SuricataSSH, SuricataFile)        │
    │       ▼                                                     │
    └─────────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                2. SESSION AGGREGATION                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  SessionManager.process_event()                             │
    │       │                                                     │
    │       │ ─── Correlate events by flow_id                     │
    │       │ ─── Aggregate HTTP, DNS, TLS, SSH data              │
    │       │ ─── Track packet/byte counters                      │
    │       │ ─── Manage session lifecycle (timeout: 2 min)       │
    │       │                                                     │
    │       ▼                                                     │
    │  SuricataSession (Enriched Session Object)                  │
    │       │                                                     │
    │       │ Contains:                                           │
    │       │  • Flow metrics (packets, bytes, duration)          │
    │       │  • HTTP data (methods, URIs, status codes)          │
    │       │  • DNS data (queries, answers)                      │
    │       │  • TLS data (SNI, versions, certificates)           │
    │       │  • SSH data (client/server versions)                │
    │       ▼                                                     │
    └─────────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              3. FEATURE EXTRACTION                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  AdaptiveFlowFeatureExtractor.extract_from_flow()           │
    │       │                                                     │
    │       │ ─── Auto-discover field paths                       │
    │       │ ─── Extract 9 core features:                        │
    │       │     1. dest_port                                    │
    │       │     2. duration                                     │
    │       │     3. total_fwd_packets                            │
    │       │     4. total_bwd_packets                            │
    │       │     5. total_fwd_bytes                              │
    │       │     6. total_bwd_bytes                              │
    │       │     7. flow_bytes_per_sec (derived)                 │
    │       │     8. flow_packets_per_sec (derived)               │
    │       │     9. down_up_ratio (derived)                      │
    │       │                                                     │
    │       │ ─── Extract application layer features (optional)   │
    │       ▼                                                     │
    │  pandas.DataFrame (Feature Vector)                          │
    │       │                                                     │
    └─────────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │               4. ANOMALY DETECTION                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │         AnomalyDetector.detect_anomalies()            │  │
    │  │                        │                               │  │
    │  │    ┌──────────────────┴──────────────────┐            │  │
    │  │    ▼                                     ▼            │  │
    │  │  detect_ml_anomaly()          detect_statistical_anomaly()│
    │  │    │                                     │            │  │
    │  │    │ 1. StandardScaler.transform()       │ 1. Z-Score  │  │
    │  │    │ 2. DT.predict() → 20%               │ 2. IQR      │  │
    │  │    │ 3. RF.predict() → 30%               │ 3. Feature  │  │
    │  │    │ 4. XGB.predict() → 50%              │    weights  │  │
    │  │    │ 5. Voting (≥2/3)                    │            │  │
    │  │    │                                     │            │  │
    │  │    └──────────────────┬──────────────────┘            │  │
    │  │                       ▼                               │  │
    │  │             combined_score = max(ml, stat)            │  │
    │  └───────────────────────────────────────────────────────┘  │
    │                          │                                  │
    └─────────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              5. BEHAVIORAL ANALYSIS                         │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  BehavioralAnalyzer.process_session()                       │
    │       │                                                     │
    │       │ ─── Track per-IP behavior over time window (5 min) │
    │       │ ─── Calculate anomaly scores:                       │
    │       │     • Port Scan Score (many ports / few IPs)        │
    │       │     • Host Scan Score (many IPs / few ports)        │
    │       │     • Brute Force Score (auth failures > 3)         │
    │       │     • Volume Anomaly Score (>1MB/s or >1000 pps)    │
    │       │                                                     │
    │       │ ─── Threshold: overall_anomaly_score > 0.5          │
    │       ▼                                                     │
    │  Behavioral Features Dict                                   │
    │       │                                                     │
    └─────────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                6. FLOW FINALIZATION                         │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  FlowFinalizer.process_session()                            │
    │       │                                                     │
    │       │ ─── Handle zero-byte flow patterns                  │
    │       │     (detect brute force: >1 flow/sec to same dest)  │
    │       │ ─── Combine ML + Statistical + Behavioral results   │
    │       │ ─── Generate alerts if is_anomalous = True          │
    │       │ ─── Save results to CSV                             │
    │       ▼                                                     │
    └─────────────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  7. ALERT OUTPUT                            │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  HybridNIDS.handle_alert()                                  │
    │       │                                                     │
    │       ├── Console logging (hybrid-nids logger)              │
    │       ├── File logging (hybrid_nids.log)                    │
    │       ├── Output file (if specified)                        │
    │       ├── Telegram notification (if enabled)                │
    │       └── CSV export (flow_results.csv)                     │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

---

## 📦 Chi Tiết Các Thành Phần Python & ML

### 1. Data Pipeline & Preprocessing

#### 📁 `hybrid_nids.py` - Main Orchestrator

```python
# Core training function
def _load_and_preprocess_dataset(self, dataset_path):
    """
    Tiền xử lý dataset CICIDS2017:
    - Load từ directory (multiple CSV) hoặc single file
    - Clean column names (strip whitespace)
    - Replace inf values → NaN
    - Fill NaN với median (numeric columns)
    - Drop duplicates
    - Convert labels: 'BENIGN' → 0, others → 1
    """
```

**Feature Mapping (CICIDS2017 → Suricata):**

| CICIDS2017 Column | Suricata Feature |
|-------------------|------------------|
| Destination Port | dest_port |
| Flow Duration | duration |
| Total Fwd Packets | total_fwd_packets |
| Total Backward Packets | total_bwd_packets |
| Total Length of Fwd Packets | total_fwd_bytes |
| Total Length of Bwd Packets | total_bwd_bytes |
| Flow Bytes/s | flow_bytes_per_sec |
| Flow Packets/s | flow_packets_per_sec |
| Down/Up Ratio | down_up_ratio |

#### 📁 `utils/dataset_balancer.py` - Data Balancing

```python
def integrate_binary_balancing(df, target_col='Label', benign_value=0):
    """
    Cân bằng dataset cho binary classification:
    - Separate benign vs attack samples
    - Undersample majority class
    - Sample size = min(benign_count, attack_count)
    - Random shuffle with seed=42
    """
```

---

### 2. Feature Engineering

#### 📁 `utils/adaptive_flow_features.py` - Feature Extractor

**Core Features (9 features đã được train):**

```python
ALIGNED_FEATURES = [
    "dest_port",          # Destination Port
    "duration",           # Flow Duration
    "total_fwd_packets",  # Total Fwd Packets
    "total_bwd_packets",  # Total Backward Packets
    "total_fwd_bytes",    # Total Length of Fwd Packets
    "total_bwd_bytes",    # Total Length of Bwd Packets
    "flow_bytes_per_sec", # Flow Bytes/s (derived)
    "flow_packets_per_sec", # Flow Packets/s (derived)
    "down_up_ratio"       # Down/Up Ratio (derived)
]
```

**Adaptive Field Path Discovery:**

```python
# Tự động phát hiện đường dẫn field trong Suricata logs
field_paths = {
    'dest_port': ['dport', 'dest_port'],
    'duration': ['dur', 'duration', 'flow.duration'],
    'total_fwd_packets': ['spkts', 'pkts_toserver', 'total_fwd_packets'],
    'total_bwd_packets': ['dpkts', 'pkts_toclient', 'total_bwd_packets'],
    'total_fwd_bytes': ['sbytes', 'bytes_toserver', 'total_fwd_bytes'],
    'total_bwd_bytes': ['dbytes', 'bytes_toclient', 'total_bwd_bytes']
}
```

**Application Layer Features (từ enriched sessions):**

| Category | Features |
|----------|----------|
| **HTTP** | http_event_count, has_http, has_http_get, has_http_post, http_method_count, has_http_error, has_http_auth_error |
| **DNS** | dns_event_count, has_dns, dns_query_count, dns_answer_count, has_dns_failure |
| **TLS** | tls_event_count, has_tls, tls_sni_count |
| **SSH** | ssh_event_count, has_ssh |
| **State** | is_rejected, is_established, is_closed, is_reset |
| **Combined** | app_layer_count, has_mixed_protocols |

---

### 3. Machine Learning Models

#### Training Pipeline (`hybrid_nids.py`)

```python
# 1. DECISION TREE
dt = DecisionTreeClassifier(
    max_depth=10, 
    random_state=42, 
    class_weight='balanced'
)

# 2. RANDOM FOREST
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced'
)

# 3. XGBOOST với Hyperparameter Tuning
param_grid = {
    'max_depth': [6, 9, 12],
    'n_estimators': [100, 150],
    'learning_rate': [0.05, 0.1, 0.2]
}
grid_search = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid, cv=3, scoring='f1_macro'
)
```

**Model Weight trong Ensemble:**

| Model | Weight | Role |
|-------|--------|------|
| Decision Tree | 20% | Fast baseline detection |
| Random Forest | 30% | Robust ensemble |
| XGBoost | 50% | High accuracy, main detector |

**Voting Logic:**
- 3 models available: anomaly = (votes ≥ 2)
- 2 models available: anomaly = (votes ≥ 1)

---

### 4. Session Management

#### 📁 `utils/session_manager.py`

**SuricataSession Dataclass:**

```python
@dataclass
class SuricataSession:
    # Identifiers
    flow_id: str
    saddr: str, sport: str, daddr: str, dport: str, proto: str
    
    # Flow metrics
    starttime, endtime, duration, state, appproto
    
    # Packet/byte counters
    total_fwd_packets, total_bwd_packets
    total_fwd_bytes, total_bwd_bytes
    
    # HTTP data
    http_methods: List[str]
    http_status_codes: List[str]
    http_hosts, http_uris, http_user_agents: List[str]
    
    # DNS data
    dns_queries, dns_answers, dns_rrtypes: List[str]
    
    # TLS data
    tls_sni, tls_versions, tls_subjects, tls_issuers: List[str]
    
    # SSH data
    ssh_client_versions, ssh_server_versions: List[str]
    
    # Derived metrics (calculated on finalize)
    flow_bytes_per_sec, flow_packets_per_sec, down_up_ratio: float
```

**Session Lifecycle:**

```
New Event → SessionManager.process_event()
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
SuricataFlow   SuricataHTTP   SuricataDNS ...
    │               │               │
    └───────────────┼───────────────┘
                    ▼
        Update SuricataSession
                    │
    ┌───────────────┴───────────────┐
    ▼                               ▼
Flow State in                  Session Timeout
['closed','established',       (120 seconds)
 'fin','rst']                       │
    │                               │
    └───────────────┬───────────────┘
                    ▼
           session.finalize()
                    │
                    ▼
         Finalized Session → Detection
```

---

### 5. Behavioral Analysis

#### 📁 `utils/behavioral_analyzer.py`

**IPBehavior Tracking (per IP address):**

```python
@dataclass
class IPBehavior:
    # Time window: 5 minutes (300 seconds)
    window_size: int = 300
    
    # Flow tracking
    total_flows, active_flows, completed_flows: int
    
    # Destination diversity
    dest_ips: Set[str]
    dest_ports: Set[int]
    unique_dest_ips_window: List[Tuple[timestamp, ip]]
    unique_dest_ports_window: List[Tuple[timestamp, port]]
    
    # Volume tracking
    bytes_sent_window: List[Tuple[timestamp, bytes]]
    packets_sent_window: List[Tuple[timestamp, packets]]
    
    # Application metrics
    http_requests, http_errors: int
    dns_queries, dns_failures: int
    tls_handshakes, tls_failures: int
    ssh_attempts: int
    
    # Auth failure tracking
    failed_auth_count: int
    auth_attempt_window: List[Tuple[timestamp, service, count]]
    
    # Anomaly scores (0.0 → 1.0)
    port_scan_score: float
    host_scan_score: float
    brute_force_anomaly_score: float
    volume_anomaly_score: float
    overall_anomaly_score: float
```

**Attack Detection Algorithms:**

| Attack Type | Detection Logic | Threshold |
|-------------|-----------------|-----------|
| **Port Scan** | ports_in_window / ips_in_window > 10 | score > 0.7 |
| **Host Scan** | ips_in_window / ports_in_window > 5 | score > 0.7 |
| **Brute Force** | auth_failures to same service > 3 | score > 0.7 |
| **Volume Anomaly** | bytes/sec > 1MB or packets/sec > 1000 | score > 0.7 |

---

### 6. Anomaly Detection Engine

#### 📁 `utils/anomaly_detector.py`

**ML Detection:**

```python
def detect_ml_anomaly(self, features: pd.DataFrame) -> Dict:
    # 1. Extract base features
    # 2. Scale with StandardScaler
    # 3. Get predictions from each model
    dt_pred = dt_model.predict(features_scaled)
    rf_pred = rf_model.predict(features_scaled)
    xgb_pred = xgb_model.predict(features_scaled)
    
    # 4. Get probability scores
    dt_score = dt_model.predict_proba(features_scaled)[0][1]
    rf_score = rf_model.predict_proba(features_scaled)[0][1]
    xgb_score = xgb_model.predict_proba(features_scaled)[0][1]
    
    # 5. Weighted combination
    combined_score = (dt_score * 0.2 + rf_score * 0.3 + xgb_score * 0.5)
    
    # 6. Voting
    votes = int(dt_pred) + int(rf_pred) + int(xgb_pred)
    is_anomalous = votes >= 2
    
    return {
        'is_anomalous': is_anomalous,
        'score': combined_score,
        'dt_prediction': dt_pred,
        'rf_prediction': rf_pred,
        'xgb_prediction': xgb_pred
    }
```

**Statistical Detection:**

```python
def detect_statistical_anomaly(self, features: pd.DataFrame) -> Dict:
    for feature in base_features.columns:
        stats = self.baseline[feature]
        
        # Z-Score analysis
        z_score = (value - stats['mean']) / stats['std']
        
        # IQR outlier detection
        iqr = stats['iqr']
        lower_bound = stats['q1'] - 1.5 * iqr
        upper_bound = stats['q3'] + 1.5 * iqr
        is_outlier = value < lower_bound or value > upper_bound
        
        # Conservative threshold: z > 4.0
        if abs(z_score) > 4.0 or is_outlier:
            anomaly_score += abs(z_score) * feature_weight
    
    # Require at least 2 anomalous features
    is_anomalous = anomaly_score > 0.7 or high_z_features >= 2
```

**Baseline Statistics (created from benign traffic):**

```json
{
    "feature_name": {
        "mean": float,
        "std": float,
        "min": float,
        "max": float,
        "q1": float,
        "median": float,
        "q3": float,
        "iqr": float
    }
}
```

---

### 7. Flow Finalizer

#### 📁 `utils/flow_finalizer.py`

**Zero-byte Pattern Detection (Brute Force):**

```python
def _handle_zero_byte_flow(self, session_dict):
    """
    Detect brute force attacks via zero-byte flow patterns:
    - Track flows with 0 bytes per (src_ip, dst_ip:port:proto)
    - If count >= threshold (3) AND rate > 1 flow/sec:
      → Flag as suspicious brute force pattern
    """
    dest_key = f"{dst_ip}:{dst_port}:{proto}"
    
    if zero_byte_flows[src_ip][dest_key]['count'] >= threshold:
        rate = count / time_span
        if rate > 1.0:  # More than 1 flow per second
            return {
                'is_anomalous': True,
                'zero_byte_pattern': True,
                'zero_byte_rate': rate
            }
```

**Detection Result Structure:**

```python
result = {
    'flow_id': str,
    'timestamp': float,
    'src_ip': str,
    'dst_ip': str,
    'dst_port': str,
    'proto': str,
    'app_proto': str,
    'duration': float,
    'total_bytes': int,
    'total_packets': int,
    'ml_result': {
        'is_anomalous': bool,
        'score': float,
        'dt_prediction': int,
        'rf_prediction': int,
        'xgb_prediction': int
    },
    'stat_result': {
        'is_anomalous': bool,
        'score': float,
        'details': List[feature_anomalies]
    },
    'combined_score': float,
    'is_anomalous': bool,
    'behavioral_features': Optional[Dict],
    'session': Dict  # Full session data
}
```

---

## 📊 Dataset & Training

### CICIDS2017 Dataset

**Files Used:**
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
- Friday-WorkingHours-Morning.pcap_ISCX.csv
- Monday-WorkingHours.pcap_ISCX.csv
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- Tuesday-WorkingHours.pcap_ISCX.csv
- Wednesday-workingHours.pcap_ISCX.csv

**Training Process:**

```bash
python hybrid_nids.py --train /path/to/CICIDS2017/ --model_dir ./model
```

**Output Files:**
- `model/dt_model.pkl` - Decision Tree model
- `model/rf_model.pkl` - Random Forest model
- `model/xgb_model.pkl` - XGBoost model
- `model/scaler.pkl` - StandardScaler
- `model/label_encoder.pkl` - LabelEncoder
- `model/baseline.json` - Statistical baseline
- `model/features.json` - Feature list

---

## 🚀 Cài Đặt & Sử Dụng

### Prerequisites

- Python 3.8+
- Suricata IDS
- Telegram Bot (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/Matth3wV0/A-Suricata-and-Machine-Learning-Based-Hybrid-Network-Intrusion-Detection-System
cd A-Suricata-and-Machine-Learning-Based-Hybrid-Network-Intrusion-Detection-System

# Install dependencies
pip install -r requirements.txt

# Create model directory
mkdir -p ./model
```

### Configuration

```bash
# .env file (for Telegram alerts)
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### Usage

```bash
# Train models
python hybrid_nids.py --train /path/to/dataset --model_dir ./model

# Real-time monitoring with Telegram alerts
python hybrid_nids.py --realtime /var/log/suricata/eve.json --telegram

# Analyze existing logs
python hybrid_nids.py --analyze /path/to/eve.json --output results.txt

# Interactive training with Jupyter
jupyter notebook training.ipynb
```

---

## 📁 Cấu Trúc Project

```
.
├── hybrid_nids.py              # Main system orchestrator
├── training.ipynb              # Interactive training notebook
│
├── utils/                      # Core utility modules
│   ├── adaptive_flow_features.py   # Feature extraction
│   ├── anomaly_detector.py         # ML + Statistical detection
│   ├── behavioral_analyzer.py      # IP behavior tracking
│   ├── dataset_balancer.py         # Data balancing
│   ├── flow_finalizer.py           # Session finalization
│   ��── session_manager.py          # Session aggregation
│   └── service_whitelist.py        # Service whitelisting
│
├── suricata/                   # Suricata integration
│   ├── suricata_parser.py          # JSON log parser
│   └── suricata_flows.py           # Dataclass definitions
│
├── telegram_module/            # Alerting
│   └── telegram_alert.py           # Telegram integration
│
├── model/                      # Trained models (generated)
│   ├── dt_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── baseline.json
│   └── features.json
│
├── requirements.txt            # Dependencies
├── .env                        # Configuration
└── README.md                   # This file
```

---

## 📚 Key Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | - | Data manipulation |
| numpy | - | Numerical operations |
| scikit-learn | - | ML models (DT, RF), preprocessing |
| xgboost | - | XGBoost classifier |
| python-dotenv | - | Environment variables |
| python-dateutil | - | Timestamp parsing |
| python-telegram-bot | - | Telegram alerts |

---

## 🙏 Acknowledgments

- CICIDS2017 dataset creators (Canadian Institute for Cybersecurity)
- Suricata team for their excellent IDS engine
- Authors of "A Suricata and Machine Learning Based Hybrid Network Intrusion Detection System"

---

## 👤 Author

**Created with ❤️ by M4tth3wV0**

*Responsible for: Python development, Machine Learning pipeline, Feature engineering, Anomaly detection algorithms, System integration*
