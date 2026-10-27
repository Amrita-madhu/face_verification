import joblib

model = joblib.load("results/hybrid_model.joblib")

print("Model type:", type(model))

if isinstance(model, dict):
    print("\nSaved components:")
    for key, value in model.items():
        print(key, ":", type(value))
else:
    print("\nModel attributes:")
    print([name for name in dir(model) if not name.startswith("_")])