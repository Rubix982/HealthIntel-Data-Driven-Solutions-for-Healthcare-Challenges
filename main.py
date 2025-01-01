import pandas
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


class Constants:

    class DataTypes:

        NUMBER: str = "number"
        OBJECT: str = "object"
        DATETIME: str = "datetime"

    class AnalysisKeys:

        CLUSTER: str = "Cluster"
        AGE_DISTRIBUTION: str = "Age Distribution"
        GENDER_DISTRIBUTION: str = "Gender Distribution"
        GEOGRAPHY_DISTRIBUTION: str = "Geography Distribution"
        MOST_FREQUENT_DISEASES: str = "Most Frequent Diseases"
        CORPORATE_CLIENT___OWN_SELF: str = "Corporate Client / Own Self"

    class Visualization:

        CMAP_VIRIDIS: str = "viridis"

    ERRORS_COERCE: str = "coerc"


class Columns:

    AGE: str = "Age"
    CITY: str = "City"
    GENDER: str = "Gender"
    HOSPITAL: str = "Hospital"
    PATIENT_ID: str = "Patient ID"
    ADMIT_DATE: str = "Admit Date"
    DOCTOR_NAME: str = "Doctor Name"
    DISCHARGE_DATE: str = "Discharge Date"
    LENGTH_OF_STAY: str = "Length of Stay"
    DISEASE_DIAGNOSED: str = "Disease Diagnosed"


def analyze_column_statistics(
    dataframe: pandas.DataFrame,
) -> Union[pandas.Series, pandas.Series, dict[str, tuple]]:

    # Numeric columns - descriptive statistics

    numeric_columns = dataframe.select_dtypes(
        include=[Constants.DataTypes.NUMBER]
    ).columns
    numeric_stats: pandas.Series = dataframe[numeric_columns].describe()

    # Categorical columns - frequency counts
    categorical_columns = dataframe.select_dtypes(
        include=[Constants.DataTypes.OBJECT]
    ).columns
    categorical_stats: pandas.Series = dataframe[categorical_columns].describe()

    # Date columns - time range statistics
    dataframe[Columns.ADMIT_DATE] = pandas.to_datetime(
        dataframe[Columns.ADMIT_DATE], errors=Constants.ERRORS_COERCE
    )
    dataframe[Columns.DISCHARGE_DATE] = pandas.to_datetime(
        dataframe[Columns.DISCHARGE_DATE], errors=Constants.ERRORS_COERCE
    )

    # Calculate Length of Stay
    dataframe[Columns.LENGTH_OF_STAY] = (
        dataframe[Columns.DISCHARGE_DATE] - dataframe[Columns.ADMIT_DATE]
    ).dt.days

    # Handle missing values for length of stay and dates
    data_columns = dataframe.select_dtypes(
        include=[Constants.DataTypes.DATETIME]
    ).columns
    date_stats: dict[str, tuple] = {
        col: (dataframe[col].min(), dataframe[col].max()) for col in data_columns
    }

    return numeric_stats, categorical_stats, date_stats


def analyze_demographics(dataframe: pandas.DataFrame) -> dict:
    demographics: dict = {
        Constants.AnalysisKeys.AGE_DISTRIBUTION: dataframe[Columns.AGE].describe(),
        Constants.AnalysisKeys.GENDER_DISTRIBUTION: dataframe[
            Columns.GENDER
        ].value_counts(),
        Constants.AnalysisKeys.GEOGRAPHY_DISTRIBUTION: dataframe[
            Columns.CITY
        ].value_counts(),
    }
    return demographics


def analyze_disease_treatment(dataframe: pandas.DataFrame) -> dict:
    disease_treatment_analysis = {
        Constants.AnalysisKeys.MOST_FREQUENT_DISEASES: dataframe[
            Columns.DISEASE_DIAGNOSED
        ]
        .value_counts()
        .head(),
    }
    return disease_treatment_analysis


def analyze_resource_allocation(dataframe: pandas.DataFrame) -> pandas.Series[int]:
    hospital_volume: pandas.Series[int] = dataframe[Columns.HOSPITAL].value_counts()
    return hospital_volume


def analyze_funding_type(dataframe: pandas.DataFrame) -> pandas.Series[int]:
    funding_analysis: pandas.Series[int] = dataframe[
        Constants.AnalysisKeys.CORPORATE_CLIENT___OWN_SELF
    ].value_counts()
    return funding_analysis


# Length of Stay Analysis
def analyze_los(dataframe: pandas.DataFrame) -> pandas.Series[any]:
    # Ensure that missing Length of Stay values do not cause errors
    los_analysis: pandas.Series[any] = (
        dataframe.groupby(Columns.DISEASE_DIAGNOSED)[Columns.LENGTH_OF_STAY]
        .mean()
        .dropna()
    )
    return los_analysis


# Doctor Workload Analysis (Patient Count per Doctor)
def analyze_doctor_workload(dataframe: pandas.DataFrame):
    doctor_workload = dataframe.groupby(Columns.DOCTOR_NAME)[Columns.PATIENT_ID].count()
    return doctor_workload


# Machine learning models to analyze Length of Stay (LoS) prediction
def analyze_los_prediction(dataframe: pandas.DataFrame):
    # Preprocessing
    dataframe = dataframe.dropna(
        subset=[Columns.LENGTH_OF_STAY]
    )  # Drop rows with missing LoS
    label_encoder = LabelEncoder()
    dataframe[Columns.GENDER] = label_encoder.fit_transform(dataframe[Columns.GENDER])
    dataframe[Columns.DISEASE_DIAGNOSED] = label_encoder.fit_transform(
        dataframe[Columns.DISEASE_DIAGNOSED]
    )
    dataframe[Columns.HOSPITAL] = label_encoder.fit_transform(
        dataframe[Columns.HOSPITAL]
    )

    # Features and target
    X = dataframe[
        [Columns.AGE, Columns.GENDER, Columns.DISEASE_DIAGNOSED, Columns.HOSPITAL]
    ]  # Add more features if necessary
    y = dataframe[Columns.LENGTH_OF_STAY]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardizing features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Regressor
    model = RandomForestRegressor()
    model.fit(X_train_scaled, y_train)

    # Predictions and Evaluation
    y_pred = model.predict(X_test_scaled)
    print(
        "Length of Stay Prediction Mean Absolute Error:",
        mean_absolute_error(y_test, y_pred),
    )


# Machine learning models to predict Funding Type
def analyze_funding_type_prediction(dataframe: pandas.DataFrame):
    # Preprocessing
    label_encoder = LabelEncoder()
    dataframe[Columns.GENDER] = label_encoder.fit_transform(dataframe[Columns.GENDER])
    dataframe[Columns.DISEASE_DIAGNOSED] = label_encoder.fit_transform(
        dataframe[Columns.DISEASE_DIAGNOSED]
    )
    dataframe[Columns.HOSPITAL] = label_encoder.fit_transform(
        dataframe[Columns.HOSPITAL]
    )

    # Features and target
    X = dataframe[
        [Columns.AGE, Columns.GENDER, Columns.DISEASE_DIAGNOSED, Columns.HOSPITAL]
    ]  # Add more features if necessary
    y = label_encoder.fit_transform(
        dataframe[Constants.AnalysisKeys.CORPORATE_CLIENT___OWN_SELF]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    print(
        "Funding Type Prediction Classification Report:\n",
        classification_report(y_test, y_pred),
    )


# K-Means clustering to segment patients
def analyze_patient_clustering(dataframe: pandas.DataFrame):
    # Preprocessing for clustering
    label_encoder = LabelEncoder()
    dataframe[Columns.GENDER] = label_encoder.fit_transform(dataframe[Columns.GENDER])
    dataframe[Columns.DISEASE_DIAGNOSED] = label_encoder.fit_transform(
        dataframe[Columns.DISEASE_DIAGNOSED]
    )

    # Features for clustering
    X = dataframe[
        [Columns.AGE, Columns.GENDER, Columns.DISEASE_DIAGNOSED]
    ]  # Modify with relevant features

    # Apply KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    dataframe[Constants.CLUSTER] = kmeans.fit_predict(X)

    # Plot the clusters
    plt.scatter(
        dataframe[Columns.AGE],
        dataframe[Columns.GENDER],
        c=dataframe[Constants.CLUSTER],
        cmap=Constants.Visualization.CMAP_VIRIDIS,
    )
    plt.xlabel(Columns.AGE)
    plt.ylabel(Columns.GENDER)
    plt.title("Clustering Patients")
    plt.show()


def min_max_error_evaluation(dataframe: pandas.DataFrame):
    # Step 1: Data Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(dataframe[[Columns.LENGTH_OF_STAY]].values)

    # X will contain the sequence of previous LOS values
    # y will contain the next day's LOS (predicting the next value)
    X = (
        torch.tensor(data_scaled[:-1]).float().view(-1, 1, 1)
    )  # 3D shape: [samples, timesteps, features]
    y = torch.tensor(data_scaled[1:]).float()

    # Step 2: Define the LSTM Model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])  # Only take the last output from the LSTM
            return out

    # Step 3: Initialize Model
    model = LSTMModel(input_size=1, hidden_size=64, output_size=1)

    # Step 4: Set up Training Parameters
    criterion = nn.MSELoss()  # Mean Squared Error loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Step 5: Train the Model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Step 6: Predicting and Inverse Scaling
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        predicted = model(X)

    # Convert predictions back to original scale
    predicted_values = scaler.inverse_transform(predicted.numpy())
    actual_values = scaler.inverse_transform(y.numpy())

    # Step 7: Evaluate the Model
    mae = mean_absolute_error(actual_values, predicted_values)
    print(f"Mean Absolute Error: {mae:.4f}")

    plt.plot(actual_values, label="Actual LOS")
    plt.plot(predicted_values, label="Predicted LOS")
    plt.legend()
    plt.show()


def main():
    file_path = "data/HealthcareData.csv"
    dataframe: pandas.DataFrame = pandas.read_csv(file_path)

    # Get all column names
    column_names = dataframe.columns
    print(f"Columns: {column_names}")

    numeric_stats, categorical_stats, date_stats = analyze_column_statistics(dataframe)
    print("Numeric Statistics:\n", numeric_stats)
    print("\nCategorical Statistics:\n", categorical_stats)
    print("\nDate Statistics:\n", date_stats)

    demographics = analyze_demographics(dataframe=dataframe)
    print("\nDemographics Analysis:", demographics)

    disease_treatment = analyze_disease_treatment(dataframe=dataframe)
    print("\nDisease and Treatment Analysis:", disease_treatment)

    resource_allocation = analyze_resource_allocation(dataframe=dataframe)
    print("\nResource Allocation:", resource_allocation)

    funding_type = analyze_funding_type(dataframe=dataframe)
    print("\nFunding Type Analysis:", funding_type)

    doctor_workload = analyze_doctor_workload(dataframe=dataframe)
    print("\nDoctor Workload Analysis:", doctor_workload)

    los_analysis = analyze_los(dataframe=dataframe)
    print("\nLength of Stay Analysis:", los_analysis)

    # ML analysis
    analyze_los_prediction(dataframe=dataframe)
    analyze_funding_type_prediction(dataframe=dataframe)
    analyze_patient_clustering(dataframe=dataframe)
    min_max_error_evaluation(dataframe=dataframe)


if __name__ == "__main__":
    main()
