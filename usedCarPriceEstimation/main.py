import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("used_cars.csv")

# Preprocessing and cleaning
df["price"] = df["price"].replace(r"[\$,]", "", regex=True).astype(float)
df["milage"] = df["milage"].replace(r"[, mi.]", "", regex=True).astype(float)
df["model_year"] = df["model_year"].astype(float)

# Handling missing values
df["fuel_type"] = df["fuel_type"].fillna("Unknown")
df["accident"] = df["accident"].fillna("Unknown")
df["clean_title"] = df["clean_title"].fillna("No")

# Filter price outliers
df = df[df["price"] < 200000]


# Function to handle rare categories
def group_rare_values(dataframe, column, min_count=10):
    counts = dataframe[column].value_counts()
    rare_values = counts[counts < min_count].index
    dataframe[column] = dataframe[column].replace(rare_values, "Other")


categorical_columns = [
    "brand", "model", "fuel_type", "engine", "transmission",
    "ext_col", "int_col", "accident", "clean_title"
]

for column in categorical_columns:
    group_rare_values(df, column)

# Separate features and target
X = df.drop("price", axis=1)
y = df["price"]

# Encoding
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Train-Val-Test Split (70-15-15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("Data shapes:")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)


# Model initialization function
def create_ann_model(input_size, layers, learning_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_size,)))

    for neurons in layers:
        model.add(tf.keras.layers.Dense(neurons, activation="relu"))

    model.add(tf.keras.layers.Dense(1, activation="linear"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model


# Grid Search Configurations
configs = [
    {"name": "ANN_1", "layers": [32, 16], "batch_size": 32, "epochs": 100, "learning_rate": 0.001},
    {"name": "ANN_2", "layers": [64, 32], "batch_size": 32, "epochs": 100, "learning_rate": 0.001},
    {"name": "ANN_3", "layers": [128, 64], "batch_size": 32, "epochs": 100, "learning_rate": 0.001},
    {"name": "ANN_4", "layers": [128, 64, 32], "batch_size": 32, "epochs": 100, "learning_rate": 0.001},
    {"name": "ANN_5", "layers": [128, 64, 32], "batch_size": 64, "epochs": 100, "learning_rate": 0.001},
    {"name": "ANN_6", "layers": [128, 64, 32], "batch_size": 32, "epochs": 150, "learning_rate": 0.001},
    {"name": "ANN_7", "layers": [256, 128, 64], "batch_size": 32, "epochs": 100, "learning_rate": 0.001},
    {"name": "ANN_8", "layers": [128, 64, 32], "batch_size": 32, "epochs": 100, "learning_rate": 0.0005},
]

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

results = []
trained_models = {}

# Experiment Loop
for config in configs:
    print(f"Training {config['name']}...")

    model = create_ann_model(
        input_size=X_train.shape[1],
        layers=config["layers"],
        learning_rate=config["learning_rate"]
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[early_stop],
        verbose=0
    )

    trained_models[config["name"]] = model

    # Validation evaluation
    y_val_pred = model.predict(X_val).flatten()

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    results.append({
        "Model": config["name"],
        "Layers": str(config["layers"]),
        "Batch Size": config["batch_size"],
        "Epochs": config["epochs"],
        "Learning Rate": config["learning_rate"],
        "Val_MAE": val_mae,
        "Val_RMSE": val_rmse,
        "Val_R2": val_r2
    })

results_df = pd.DataFrame(results)
print("\nValidation Results:")
print(results_df.to_string(index=False))

# Select best architecture
best_row = results_df.sort_values(by="Val_R2", ascending=False).iloc[0]
best_model_name = best_row["Model"]

print(f"\nBest model found: {best_model_name}")
print(best_row)

# Final Evaluation on Test Set
print(f"\nEvaluating {best_model_name} on independent test set...")
best_model = trained_models[best_model_name]
y_test_pred = best_model.predict(X_test).flatten()

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print(f"Final Test MAE : {test_mae:.2f}")
print(f"Final Test RMSE: {test_rmse:.2f}")
print(f"Final Test R2   : {test_r2:.4f}")