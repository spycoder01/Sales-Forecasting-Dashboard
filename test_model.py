import joblib

model = joblib.load("xgboost_model.pkl")

print(type(model))