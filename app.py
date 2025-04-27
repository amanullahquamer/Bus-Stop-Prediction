# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # for loading .pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. DATA LOADING & PREPROCESSING
# -------------------------------
@st.cache_data
def load_data(nrows=200000):
    df = pd.read_csv(
        "mta_1708.csv",
        nrows=nrows,
        on_bad_lines="skip",
        engine="python"
    )
    # parse dates
    df["RecordedAtTime"]   = pd.to_datetime(df["RecordedAtTime"], errors="coerce")
    df["ExpectedArrivalTime"] = pd.to_datetime(df["ExpectedArrivalTime"], errors="coerce")
    df["ScheduledArrivalTime"] = pd.to_datetime(df["ScheduledArrivalTime"], errors="coerce")
    # drop nulls and duplicates
    req = [
        "DirectionRef","PublishedLineName",
        "VehicleLocation.Latitude","VehicleLocation.Longitude",
        "ArrivalProximityText","DistanceFromStop"
    ]
    df.dropna(subset=req, inplace=True)
    df.drop_duplicates(inplace=True)
    # encode target and categories
    df["ArrivalProximityText_enc"] = df["ArrivalProximityText"].factorize()[0]
    df["DirectionRef_enc"]         = LabelEncoder().fit_transform(df["DirectionRef"])
    df["PublishedLineName_enc"]    = LabelEncoder().fit_transform(df["PublishedLineName"])
    return df

df = load_data()

feature_list = [
    "DirectionRef_enc",
    "PublishedLineName_enc",
    "VehicleLocation.Latitude",
    "VehicleLocation.Longitude",
    "DistanceFromStop",
]
target = "ArrivalProximityText_enc"

# Split once so we can still show test accuracy
X = df[feature_list]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 2. LOAD PRETRAINED MODELS
# -------------------------------
@st.cache_resource
def load_models():
    rf = joblib.load("random_forest_model.pkl")    # your saved RandomForest pickle
    lr = joblib.load("logistic_regression_model.pkl")    # your saved LogisticRegression pickle
    return rf, lr

rf_model, lr_model = load_models()

# Compute test accuracies just once
acc_rf = accuracy_score(y_test, rf_model.predict(X_test))
acc_lr = accuracy_score(y_test, lr_model.predict(X_test))

# -------------------------------
# 3. STREAMLIT INTERFACE
# -------------------------------
st.title("🚏 Bus Route Optimization Predictor")

st.subheader("Test Accuracies")
st.write(f"**Random Forest:**       {acc_rf*100:.2f}%")
st.write(f"**Logistic Regression:** {acc_lr*100:.2f}%")
st.bar_chart({
    "Random Forest": [acc_rf*100],
    "Logistic Regression": [acc_lr*100]
})

# Sidebar inputs
st.sidebar.header("Input Features")
dir_le  = LabelEncoder().fit(df["DirectionRef"])
line_le = LabelEncoder().fit(df["PublishedLineName"])

dir_raw  = st.sidebar.selectbox("DirectionRef", df["DirectionRef"].unique())
line_raw = st.sidebar.selectbox("Route (PublishedLineName)", df["PublishedLineName"].unique())
lat_raw  = st.sidebar.number_input(
    "Latitude",
    float(df["VehicleLocation.Latitude"].min()),
    float(df["VehicleLocation.Latitude"].max()),
    float(df["VehicleLocation.Latitude"].mean())
)
lon_raw  = st.sidebar.number_input(
    "Longitude",
    float(df["VehicleLocation.Longitude"].min()),
    float(df["VehicleLocation.Longitude"].max()),
    float(df["VehicleLocation.Longitude"].mean())
)
dist_raw = st.sidebar.number_input(
    "DistanceFromStop",
    float(df["DistanceFromStop"].min()),
    float(df["DistanceFromStop"].max()),
    float(df["DistanceFromStop"].mean())
)

# Encode and predict
dir_enc  = int(dir_le.transform([dir_raw])[0])
line_enc = int(line_le.transform([line_raw])[0])
X_new = np.array([[dir_enc, line_enc, lat_raw, lon_raw, dist_raw]])

model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])
if st.sidebar.button("Predict"):
    if model_choice == "Random Forest":
        pred = rf_model.predict(X_new)[0]
    else:
        pred = lr_model.predict(X_new)[0]
    # invert factorization
    prox_labels = df["ArrivalProximityText"].unique()
    st.success(f"**Predicted Arrival Proximity:** {prox_labels[pred]}")

# Show a sample of raw data
st.subheader("Raw Data Sample")
st.dataframe(df.head(10))
