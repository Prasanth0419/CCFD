import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from twilio.rest import Client

# Load Dataset
df = pd.read_csv("creditcard.csv")

# Preprocess Data
df["Time"] = StandardScaler().fit_transform(df["Time"].values.reshape(-1, 1))
df["Amount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))

# Define Features & Labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
y_pred_if = iso_forest.fit_predict(X_test)

# Convert Predictions to Match Labels
y_pred_if = [1 if x == -1 else 0 for x in y_pred_if]

# Evaluate Model
print("Isolation Forest Performance:\n", classification_report(y_test, y_pred_if))

# Twilio Credentials (Replace with your values)
ACCOUNT_SID = "AC71aa4fa21887ae57b5667231c030623b"
AUTH_TOKEN = "6cb1e2051bccd0516f1f264d3789b371"
TWILIO_PHONE_NUMBER = "+18454151910"
RECIPIENT_PHONE_NUMBER = "+917530056060"


def send_fraud_alert(transaction_id, amount):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    message = client.messages.create(
        body=f"🚨 Fraud Alert! Suspicious transaction detected.\nTransaction ID: {transaction_id}\nAmount: ${amount}",
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )

    print(f"SMS sent! Message SID: {message.sid}")


# Send SMS for Fraudulent Transactions
for i in range(len(y_test)):
    if y_pred_if[i] == 1:  # If fraud detected
        transaction_id = i
        amount = X_test.iloc[i]['Amount']
        print(f"⚠️ Fraud detected! Transaction ID: {transaction_id}, Amount: ${amount}")
        send_fraud_alert(transaction_id, amount)
