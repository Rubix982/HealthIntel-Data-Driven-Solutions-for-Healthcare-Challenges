# Alias imports
import os
import math
import torch
import pandas
import folium
import squarify
import numpy as np
import torch.nn as nn
import seaborn as sns
import statsmodel.api as sm
import matplotlib.pyplot as plt

## Plotly imports
import plotly.express as px
import plotly.graph_objects as go

# From imports
from scipy import stats
from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from scipy.stats import chi2_contingency

## Statsmodel imports
from statsmodel.formula.api import ols
from statsmodel.stats.anova import AnovaRM

## Sklearn imports
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    classification_report,
    ConfusionMatrixDisplay,
)
from typing import Union


class Constants:

    class Paths:

        PLOT_OUTPUT_DIR: str = "images"

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


def create_output_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_plot_filepath(filename: str):
    return os.path.join(Constants.Paths.PLOT_OUTPUT_DIR, f"{filename}.png")


def save_plot(figure, filename: str):
    create_output_folder(Constants.Paths.PLOT_OUTPUT_DIR)
    figure.savefig(get_plot_filepath(filename))
    plt.close(figure)


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

    # Plot histograms for numeric columns
    plt.figure(figsize=(10, 6))
    dataframe[numeric_columns].hist(bins=20, color="c", edgecolor="black")
    plt.suptitle("Numeric Columns Distribution")
    save_plot(plt, "numeric_columns_distribution")

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

    # Plot age distribution
    plt.figure(figsize=(10, 6))
    dataframe[Columns.AGE].plot(kind="hist", bins=20, color="g", edgecolor="black")
    plt.title("Age Distribution")
    save_plot(plt, "age_distribution")

    return demographics


def analyze_disease_treatment(dataframe: pandas.DataFrame) -> dict:
    disease_treatment_analysis = {
        Constants.AnalysisKeys.MOST_FREQUENT_DISEASES: dataframe[
            Columns.DISEASE_DIAGNOSED
        ]
        .value_counts()
        .head(),
    }

    # Plot most frequent diseases
    plt.figure(figsize=(10, 6))
    dataframe[Columns.DISEASE_DIAGNOSED].value_counts().head().plot(
        kind="bar", color="b"
    )
    plt.title("Most Frequent Diseases")
    save_plot(plt, "most_frequent_diseases")

    return disease_treatment_analysis


def analyze_resource_allocation(dataframe: pandas.DataFrame) -> pandas.Series[int]:
    hospital_volume: pandas.Series[int] = dataframe[Columns.HOSPITAL].value_counts()

    # Plot hospital volume
    plt.figure(figsize=(10, 6))
    hospital_volume.plot(kind="bar", color="m")
    plt.title("Hospital Volume Distribution")
    save_plot(plt, "hospital_volume_distribution")

    return hospital_volume


def analyze_funding_type(dataframe: pandas.DataFrame) -> pandas.Series[int]:
    funding_analysis: pandas.Series[int] = dataframe[
        Constants.AnalysisKeys.CORPORATE_CLIENT___OWN_SELF
    ].value_counts()

    # Plot funding type distribution
    plt.figure(figsize=(10, 6))
    funding_analysis.plot(kind="bar", color="y")
    plt.title("Funding Type Distribution")
    save_plot(plt, "funding_type_distribution")

    return funding_analysis


# Length of Stay Analysis
def analyze_los(dataframe: pandas.DataFrame) -> pandas.Series[any]:
    # Ensure that missing Length of Stay values do not cause errors
    los_analysis: pandas.Series[any] = (
        dataframe.groupby(Columns.DISEASE_DIAGNOSED)[Columns.LENGTH_OF_STAY]
        .mean()
        .dropna()
    )

    # Plot LOS analysis by disease
    plt.figure(figsize=(10, 6))
    los_analysis.plot(kind="bar", color="r")
    plt.title("Average Length of Stay by Disease")
    save_plot(plt, "average_los_by_disease")

    return los_analysis


# Doctor Workload Analysis (Patient Count per Doctor)
def analyze_doctor_workload(dataframe: pandas.DataFrame):
    doctor_workload = dataframe.groupby(Columns.DOCTOR_NAME)[Columns.PATIENT_ID].count()

    # Plot doctor workload
    plt.figure(figsize=(10, 6))
    doctor_workload.plot(kind="bar", color="b")
    plt.title("Doctor Workload (Patient Count)")
    save_plot(plt, "doctor_workload")

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

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual LoS", color="g")
    plt.plot(y_pred, label="Predicted LoS", color="r", linestyle="--")
    plt.legend()
    plt.title("Actual vs Predicted Length of Stay")
    save_plot(plt, "actual_vs_predicted_los")


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

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    save_plot(plt, "funding_type_confusion_matrix")


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
    plt.savefig(get_plot_filepath("patient_clustering"))
    save_plot(plt, "patient_segmentation_clusters")


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


def quick_anova_analysis(dataframe: pandas.DataFrame):
    # Get the unique diseases from the "Disease" column
    unique_diseases = dataframe[Columns.DISEASE_DIAGNOSED].unique()

    # Prepare the groups for ANOVA
    groups = [
        dataframe[dataframe[Columns.DISEASE_DIAGNOSED] == disease][
            Columns.LENGTH_OF_STAY
        ]
        for disease in unique_diseases
    ]

    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    print("F-Statistic: ", f_stat)
    print("P-Value:", p_value)


def quick_two_way_anova_analysis(dataframe: pandas.DataFrame):
    # Fit the model
    model = ols(
        "Length_of_Stay ~ C(Disease_Diagnosed) + C(Gender) + C(Disease_Diagnosed):C(Gender)",
        data=dataframe,
    ).fit()

    # Perform Two-Way ANOVA
    anova_results = sm.stats.anova_lm(model, type=2)
    print("ANOVA Results:\n", anova_results)

    # Plot ANOVA results
    plt.figure(figsize=(10, 6))
    anova_results.plot(kind="bar", color="c")
    plt.title("ANOVA Results: Treatment Impact on Length of Stay")
    save_plot(plt, "anova_treatment_impact_los")


def quick_chi_square_analysis(dataframe: pandas.DataFrame):
    # Contingency table between Disease Diagnosed and Funding Type
    contingency_table = pandas.crosstab(
        dataframe[Columns.DISEASE_DIAGNOSED],
        dataframe[Constants.AnalysisKeys.CORPORATE_CLIENT___OWN_SELF],
    )

    # Perform Chi-Squared test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    # Display results
    print(f"Chi-Squared Test Statistic: {chi2_stat}")
    print(f"P-Value: {p_value}")
    print(f"Degrees of Freedom: {dof}")
    print(f"Expected Frequencies:\n{expected}")

    # Plot contingency table as heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, cmap=plt.cm.Blues, fmt="d")
    plt.title("Chi-Squared Test: Disease vs Gender Distribution")
    save_plot(plt, "chi_squared_disease_gender")


def quick_mean_absolute_error_with_ridge_analysis(dataframe: pandas.DataFrame):
    # Preprocessing
    X = dataframe[["Age", "Gender", "Disease Diagnosed", "Hospital"]]
    y = dataframe["Length_of_Stay"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Ridge Regression Model
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = ridge.predict(X_test_scaled)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (Ridge Regression): {mae}")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual LoS", color="g")
    plt.plot(y_pred, label="Predicted LoS", color="r", linestyle="--")
    plt.legend()
    plt.title("Ridge Regression: Actual vs Predicted Length of Stay")
    save_plot(plt, "ridge_regression_actual_vs_predicted_los")


def quick_pca_analysis(dataframe: pandas.DataFrame):
    # Preprocessing
    X = dataframe[
        [Columns.AGE, Columns.GENDER, Columns.DISEASE_DIAGNOSED, Columns.HOSPITAL]
    ]

    # Standardizing features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Visualize PCA components
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dataframe[Columns.DISEASE_DIAGNOSED])
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of Healthcare Data")
    save_plot(plt, "pca_patient_data")


def generate_additional_charts(dataframe: pandas.DataFrame) -> None:
    # Create output folder for plots
    create_output_folder(Constants.Paths.PLOT_OUTPUT_DIR)

    # Pie Chart: Distribution of Gender per City
    gender_city = (
        dataframe.groupby(Columns.CITY)[Columns.GENDER].value_counts().unstack()
    )
    gender_city.plot(kind="pie", subplots=True, figsize=(12, 6), autopct="%1.1f%%")
    plt.title("Gender Distribution per City")
    save_plot(plt, "gender_distribution_city")

    # Donut Chart: Breakdown of Disease Diagnosed by Gender
    disease_gender = dataframe.groupby(Columns.GENDER)[
        Columns.DISEASE_DIAGNOSED
    ].value_counts()
    disease_gender.plot.pie(autopct="%1.1f%%", figsize=(8, 8))
    circle = plt.Circle((0, 0), 0.7, color="white")
    plt.gca().add_artist(circle)
    plt.title("Disease Diagnosed by Gender")
    save_plot(plt, "disease_gender_donut")

    # Histogram: Distribution of Age with Gender Segmentation
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataframe, x=Columns.AGE, hue=Columns.GENDER, kde=True, bins=20)
    plt.title("Age Distribution by Gender")
    save_plot(plt, "age_distribution_gender")

    # Bar Chart: Average Length of Stay per Disease Diagnosed
    avg_los_disease = dataframe.groupby(Columns.DISEASE_DIAGNOSED)[
        Columns.LENGTH_OF_STAY
    ].mean()
    avg_los_disease.plot(kind="bar", color="skyblue", figsize=(10, 6))
    plt.title("Average Length of Stay per Disease Diagnosed")
    save_plot(plt, "avg_los_disease")

    # Line Chart: Monthly Admission Trends
    dataframe[Columns.ADMIT_DATE] = pandas.to_datetime(
        dataframe[Columns.ADMIT_DATE], errors=Constants.ERRORS_COERCE
    )
    monthly_admissions = (
        dataframe.set_index(Columns.ADMIT_DATE)
        .resample("M")[Columns.PATIENT_ID]
        .count()
    )
    monthly_admissions.plot(kind="line", marker="o", figsize=(10, 6), color="green")
    plt.title("Monthly Admission Trends")
    save_plot(plt, "monthly_admissions_trends")

    # Heatmap: Correlation Between Numeric Columns
    plt.figure(figsize=(10, 8))
    correlation_matrix = dataframe.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Columns")
    save_plot(plt, "correlation_heatmap")

    # Box Plot: Length of Stay by Disease Diagnosed
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=Columns.DISEASE_DIAGNOSED, y=Columns.LENGTH_OF_STAY, data=dataframe)
    plt.xticks(rotation=45)
    plt.title("Length of Stay Distribution by Disease Diagnosed")
    save_plot(plt, "length_of_stay_boxplot")

    # Scatter Plot: Age vs Length of Stay Colored by Gender
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=dataframe, x=Columns.AGE, y=Columns.LENGTH_OF_STAY, hue=Columns.GENDER
    )
    plt.title("Age vs Length of Stay by Gender")
    save_plot(plt, "age_vs_los_scatter")

    # Violin Plot: Age Distribution by Gender
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=Columns.GENDER, y=Columns.AGE, data=dataframe)
    plt.title("Age Distribution by Gender")
    save_plot(plt, "age_distribution_violin")

    # Bubble Chart: Hospital vs Patients with Length of Stay as Bubble Size
    hospital_patient_los = dataframe.groupby(Columns.HOSPITAL).agg(
        patient_count=(Columns.PATIENT_ID, "count"),
        avg_los=(Columns.LENGTH_OF_STAY, "mean"),
    )
    plt.figure(figsize=(12, 8))
    plt.scatter(
        hospital_patient_los.index,
        hospital_patient_los["patient_count"],
        s=hospital_patient_los["avg_los"] * 50,  # Scale bubble size
        alpha=0.6,
    )
    plt.xlabel("Hospital")
    plt.ylabel("Number of Patients")
    plt.title("Hospital vs Patients with Bubble Size as Length of Stay")
    save_plot(plt, "hospital_bubble_chart")

    # plot_top_diseases_by_city
    diseases_by_city = (
        dataframe.groupby([Columns.CITY, Columns.DISEASE_DIAGNOSED])[Columns.PATIENT_ID]
        .count()
        .unstack()
        .fillna(0)
    )
    diseases_by_city.plot(kind="bar", stacked=True, figsize=(12, 8), colormap="tab20")
    plt.title("Top Diseases by City")
    plt.ylabel("Number of Patients")
    plt.xticks(rotation=45)
    save_plot(plt, "top_diseases_by_city")

    # plot_avg_patient_age_by_hospital
    avg_age_hospital = dataframe.groupby(Columns.HOSPITAL)[Columns.AGE].mean()
    avg_age_hospital.plot(kind="bar", color="orchid", figsize=(10, 6))
    plt.title("Average Patient Age by Hospital")
    save_plot(plt, "avg_patient_age_hospital")

    # plot_doctor_patient_ratio
    doctor_patient_counts = dataframe.groupby(Columns.DOCTOR_NAME)[
        Columns.PATIENT_ID
    ].count()
    doctor_patient_counts.plot(kind="bar", figsize=(12, 6), color="teal")
    plt.title("Doctor-Patient Ratio")
    save_plot(plt, "doctor_patient_ratio")

    # Time-Series Analysis
    ## plot_daily_admissions_trends
    daily_admissions = dataframe.groupby(dataframe[Columns.ADMIT_DATE].dt.date)[
        Columns.PATIENT_ID
    ].count()
    daily_admissions.plot(kind="line", figsize=(12, 6), marker="o", color="blue")
    plt.title("Daily Admissions Trends")
    save_plot(plt, "daily_admissions_trends")

    ## plot_seasonal_los_trends
    dataframe["Season"] = dataframe[Columns.ADMIT_DATE].dt.month % 12 // 3 + 1
    seasonal_los = dataframe.groupby("Season")[Columns.LENGTH_OF_STAY].mean()
    seasonal_los.plot(kind="bar", color="gold", figsize=(10, 6))
    plt.title("Seasonal Length of Stay Trends")
    save_plot(plt, "seasonal_los_trends")

    ## plot_disease_trends_over_time
    monthly_disease_trends = (
        dataframe.groupby(
            [dataframe[Columns.ADMIT_DATE].dt.to_period("M"), Columns.DISEASE_DIAGNOSED]
        )
        .size()
        .unstack()
        .fillna(0)
    )
    monthly_disease_trends.plot(kind="line", figsize=(14, 8), colormap="tab10")
    plt.title("Disease Trends Over Time")
    save_plot(plt, "disease_trends_over_time")

    # Pair Plots
    ## plot_pairwise_relationships
    sns.pairplot(
        dataframe,
        vars=[Columns.AGE, Columns.LENGTH_OF_STAY],
        hue=Columns.GENDER,
        palette="husl",
    )
    plt.title("Pairwise Relationships")
    save_plot(plt, "pairwise_relationships")

    # Tree Maps
    ## plot_disease_distribution_treemap
    disease_counts = dataframe[Columns.DISEASE_DIAGNOSED].value_counts()
    squarify.plot(
        sizes=disease_counts.values,
        label=disease_counts.index,
        alpha=0.8,
        color=sns.color_palette("Paired"),
    )
    plt.title("Disease Distribution Tree Map")
    plt.axis("off")
    save_plot(plt, "disease_distribution_treemap")

    # Word Clouds
    ## plot_word_cloud
    def plot_word_cloud(column: str, title: str, filename: str) -> None:
        text = " ".join(dataframe[column].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text
        )
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        save_plot(plt, filename)

    ## generate_word_clouds
    plot_word_cloud(
        dataframe,
        Columns.DISEASE_DIAGNOSED,
        "Common Diseases Diagnosed",
        "common_diseases_wordcloud",
    )
    plot_word_cloud(
        dataframe,
        Columns.DOCTOR_NAME,
        "Frequent Doctor Names",
        "doctor_names_wordcloud",
    )

    # Clustering Visualzations
    ## plot_cluster_heatmap
    dataframe_encoded = dataframe.copy()
    label_encoder = LabelEncoder()
    for col in [Columns.GENDER, Columns.DISEASE_DIAGNOSED, Columns.HOSPITAL]:
        dataframe_encoded[col] = label_encoder.fit_transform(dataframe_encoded[col])

    cluster_corr = dataframe_encoded.corr()
    sns.heatmap(cluster_corr, annot=True, cmap="coolwarm")
    plt.title("Cluster Heatmap")
    save_plot(plt, "cluster_heatmap")

    ## plot_los_by_funding_type
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x=Constants.AnalysisKeys.CORPORATE_CLIENT___OWN_SELF,
        y=Columns.LENGTH_OF_STAY,
        data=dataframe,
    )
    plt.title("Length of Stay by Funding Type")
    save_plot(plt, "los_by_funding_type")

    # Radar Charts
    hospital_stats = dataframe.groupby(Columns.HOSPITAL).agg(
        patient_count=(Columns.PATIENT_ID, "count"),
        avg_los=(Columns.LENGTH_OF_STAY, "mean"),
    )
    categories = list(hospital_stats.columns)
    values = hospital_stats.mean(axis=0).tolist()
    values += values[:1]  # Circular for radar chart
    angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
    angles += angles[:1]

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)
    plt.title("Radar Chart: Hospital Metrics")
    save_plot(plt, "radar_chart_hospital_metrics")

    plot_admissions_by_region(dataframe=dataframe)
    plot_sankey_diagram(dataframe=dataframe)
    plot_funnel_chart(dataframe=dataframe)
    plot_animated_disease_trends(dataframe=dataframe)


def plot_admissions_by_region(dataframe):
    # Geolocate cities to get latitude and longitude
    geolocator = Nominatim(user_agent="geoapi")
    city_coords = {}
    for city in dataframe[Columns.CITY].unique():
        location = geolocator.geocode(city)
        if location:
            city_coords[city] = (location.latitude, location.longitude)

    # Prepare a map
    admission_map = folium.Map(
        location=[20, 78], zoom_start=5
    )  # Center map over an example region (India here)

    # Add markers for cities
    for city, coords in city_coords.items():
        patient_count = dataframe[dataframe[Columns.CITY] == city][
            Columns.PATIENT_ID
        ].count()
        folium.CircleMarker(
            location=coords,
            radius=patient_count / 10,  # Scale the radius
            popup=f"{city}: {patient_count} patients",
            color="blue",
            fill=True,
            fill_color="blue",
        ).add_to(admission_map)

    # Save the map as HTML
    admission_map.save(get_plot_filepath("admissions_by_region.html"))


def plot_sankey_diagram(dataframe):
    # Prepare data for the Sankey diagram
    source = []
    target = []
    value = []

    # Map city to hospital
    city_to_hospital = (
        dataframe.groupby([Columns.CITY, Columns.HOSPITAL])[Columns.PATIENT_ID]
        .count()
        .reset_index()
    )
    for _, row in city_to_hospital.iterrows():
        source.append(row[Columns.CITY])
        target.append(row[Columns.HOSPITAL])
        value.append(row[Columns.PATIENT_ID])

    # Map hospital to disease
    hospital_to_disease = (
        dataframe.groupby([Columns.HOSPITAL, Columns.DISEASE_DIAGNOSED])[
            Columns.PATIENT_ID
        ]
        .count()
        .reset_index()
    )
    for _, row in hospital_to_disease.iterrows():
        source.append(row[Columns.HOSPITAL])
        target.append(row[Columns.DISEASE_DIAGNOSED])
        value.append(row[Columns.PATIENT_ID])

    # Create Sankey diagram
    fig = go.Figure(
        go.Sankey(
            node=dict(label=list(set(source + target))),
            link=dict(
                source=[list(set(source + target)).index(src) for src in source],
                target=[list(set(source + target)).index(tgt) for tgt in target],
                value=value,
            ),
        )
    )

    fig.update_layout(
        title_text="Patient Flow: City → Hospital → Disease", font_size=10
    )
    fig.write_html(get_plot_filepath("sankey_patient_flow.html"))


def plot_funnel_chart(dataframe):
    # Prepare data for funnel
    stages = ["Admissions", "Diagnoses", "Discharges"]
    counts = [
        dataframe[Columns.ADMIT_DATE].count(),
        dataframe[Columns.DISEASE_DIAGNOSED].count(),
        dataframe[Columns.DISCHARGE_DATE].count(),
    ]

    fig = go.Figure(go.Funnel(y=stages, x=counts, textinfo="value+percent initial"))

    fig.update_layout(title_text="Patient Journey Funnel", font_size=10)
    fig.write_html(get_plot_filepath("funnel_chart.html"))


def plot_animated_disease_trends(dataframe):
    # Prepare data for animation
    dataframe[Columns.ADMIT_DATE] = pandas.to_datetime(dataframe[Columns.ADMIT_DATE])
    dataframe["YearMonth"] = dataframe[Columns.ADMIT_DATE].dt.to_period("M").astype(str)
    animated_data = (
        dataframe.groupby(["YearMonth", Columns.DISEASE_DIAGNOSED])[Columns.PATIENT_ID]
        .count()
        .reset_index()
    )

    fig = px.bar(
        animated_data,
        x=Columns.DISEASE_DIAGNOSED,
        y=Columns.PATIENT_ID,
        color=Columns.DISEASE_DIAGNOSED,
        animation_frame="YearMonth",
        title="Disease Trends Over Time",
        labels={Columns.DISEASE_DIAGNOSED: "Disease", Columns.PATIENT_ID: "Count"},
    )

    fig.write_html(get_plot_filepath("animated_disease_trends.html"))


def main():
    file_path = "data/HealthcareData.csv"
    dataframe: pandas.DataFrame = pandas.read_csv(file_path)

    # Get all column names
    print(f"Columns: {dataframe.columns}")

    generate_additional_charts(dataframe=dataframe)
    return

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

    # Statistical analysis (ANOVA, etc)
    quick_anova_analysis(dataframe=dataframe)
    quick_two_way_anova_analysis(dataframe=dataframe)
    quick_chi_square_analysis(dataframe=dataframe)
    quick_mean_absolute_error_with_ridge_analysis(dataframe=dataframe)
    quick_pca_analysis(dataframe=dataframe)


if __name__ == "__main__":
    main()
