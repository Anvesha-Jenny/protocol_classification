import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text

# --- Load dataset ---
df = pd.read_csv("protocol_dataset.csv")

# Features and labels
X = df[["Data_Rate_bps", "Payload_bits", "Noise_Level_dB", "Latency_ms", "Array_Rate_Hz"]]
y = df["Label"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Train a Random Forest Classifier ---
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully with accuracy: {accuracy*100:.2f}%")

# --- Example prediction ---
sample = np.array([[500000, 64, 20, 4, 2000000]])  # [DataRate, Payload, Noise, Latency, ArrayRate]
pred = model.predict(sample)
print(f"\nPredicted Protocol for {sample.tolist()[0]} → {pred[0]}")

# --- Extract FSM from a single decision tree ---
# Use one tree for interpretability
single_tree = model.estimators_[0]
tree_rules = export_text(single_tree, feature_names=list(X.columns))
print("\nFinite State Machine Representation (Decision Tree Rules):\n")
print(tree_rules)

# --- Simple FSM function ---
def protocol_fsm(data_rate, payload, noise, latency, array_rate):
    """Finite state machine based on general learned behavior."""
    state = "Start"
    while True:
        if state == "Start":
            if data_rate > 3e6:
                state = "HighSpeed"
            elif noise > 35:
                state = "HighNoise"
            else:
                state = "MediumRange"

        elif state == "HighSpeed":
            return "SPI"

        elif state == "HighNoise":
            return "I2C"

        elif state == "MediumRange":
            if l:
                return "UART"
            else:
                return "SPI"

# --- Test FSM ---
print("\nFSM Output Example:")
fsm_result = protocol_fsm(96000, 4, 5,23, 200)
print(f"For given parameters → Predicatency > 8ted protocol: {fsm_result}")
