import argparse
import logging
import os
from datetime import datetime
from logging import Logger

import folium
import math
import numpy as np
import openai
import pandas
import plotly.express as px
import plotly.graph_objects as go
import squarify
import statsmodels.api as sm
import torch
import torch.nn as nn
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from lifelines import KaplanMeierFitter
from openai import OpenAI
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.formula.api import ols
from wordcloud import WordCloud

# Load environment variables from .env file
load_dotenv()

# Environment Variables
OPENAI_API_KEY: str = ""
OPENAI_CLIENT: openai.Client

# Global logger variable
logger: Logger = None


class Constants:
    class Matplotlib:
        class CMapColors:
            VIRIDIS: str = "viridis"
            BLUES: str = "Blues"
            COOL_WARM: str = "coolwarm"

        class Colors:
            SKY_BLUE: str = "skyblue"

    class Paths:
        PLOT_OUTPUT_DIR: str = "images"
        LOGS_DIR: str = "logs"

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

    ERRORS_COERCE = "coerce"


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


def set_env_vars():
    # Access the API key
    global OPENAI_API_KEY
    global OPENAI_CLIENT
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)

    if not OPENAI_API_KEY:
        log_time_info("OPENAI_API_KEY is not set in the .env file.")


def create_output_folder(folder_path: str):
    if not os.path.exists(folder_path):
        logger.info(f"Creating folder: {folder_path}")
        os.makedirs(folder_path, exist_ok=True)


def get_plot_filepath(filename: str):
    filename_split = filename.split(".")
    filename = (
        f"{filename}.png"
        if len(filename_split) == 1
           or filename_split[-1] not in ["html", "png", "jpeg", "jpg", "gif"]
        else filename
    )
    return os.path.join(Constants.Paths.PLOT_OUTPUT_DIR, filename)


def save_plot(figure, filename: str):
    figure.savefig(get_plot_filepath(filename))
    plt.close(fig=figure)


def save_plot_html(figure, filename: str):
    figure.write_html(get_plot_filepath(filename))
    plt.close(fig=figure)


# Enhanced Data Cleaning Function
def clean_data(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    logger.info("Starting data cleaning process.")

    # Handling missing values
    logger.debug("Handling missing values for Age and Length of Stay.")
    dataframe.fillna({
        Columns.AGE: dataframe[Columns.AGE].median(),
        Columns.LENGTH_OF_STAY: dataframe[Columns.LENGTH_OF_STAY].median()
    }, inplace=True)

    # Remove outliers based on Z-scores
    logger.debug("Removing outliers in Length of Stay based on Z-scores.")
    z_scores = np.abs(stats.zscore(dataframe[Columns.LENGTH_OF_STAY]))
    original_size = len(dataframe)
    dataframe = dataframe[(z_scores < 3)]
    logger.info(f"Removed {original_size - len(dataframe)} outliers from the dataset.")

    logger.info("Data cleaning process completed.")
    return dataframe


# Enhanced Correlation Heatmap
def enhanced_correlation_heatmap(dataframe: pandas.DataFrame):
    plt.figure(figsize=(12, 10))
    # Select only numeric columns
    numeric_dataframe = dataframe.select_dtypes(include=['number'])

    # Compute the correlation matrix for numeric columns only
    correlation_matrix = numeric_dataframe.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Enhanced Correlation Heatmap")
    save_plot(plt.gcf(), "enhanced_correlation_heatmap")


# Enhanced Predictive Modeling with Hyperparameter Tuning
def enhanced_los_prediction(dataframe: pandas.DataFrame):
    label_encoder = LabelEncoder()
    dataframe[Columns.GENDER] = label_encoder.fit_transform(dataframe[Columns.GENDER])
    dataframe[Columns.DISEASE_DIAGNOSED] = label_encoder.fit_transform(dataframe[Columns.DISEASE_DIAGNOSED])
    dataframe[Columns.HOSPITAL] = label_encoder.fit_transform(dataframe[Columns.HOSPITAL])

    x = dataframe[[Columns.AGE, Columns.GENDER, Columns.DISEASE_DIAGNOSED, Columns.HOSPITAL]]
    y = dataframe[Columns.LENGTH_OF_STAY]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    model = GradientBoostingRegressor()
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=3)
    grid_search.fit(x_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    log_time_info(f"Enhanced Gradient Boosting Regression MAE: {mae:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual LoS", color="blue")
    plt.plot(y_pred, label="Predicted LoS", color="orange", linestyle="--")
    plt.legend()
    plt.title("Enhanced Actual vs Predicted LoS")
    save_plot(plt.gcf(), "enhanced_actual_vs_predicted_los")


# Enhanced Geographic Heatmap
def geographic_heatmap(dataframe: pandas.DataFrame):
    geolocator = Nominatim(user_agent="geoapi")
    city_coordinates = {}
    for city in dataframe[Columns.CITY].unique():
        location = geolocator.geocode(city)
        if location:
            city_coordinates[city] = (location.latitude, location.longitude)

    m = folium.Map(location=[20, 78], zoom_start=5)

    for city, coordinates in city_coordinates.items():
        count = dataframe[dataframe[Columns.CITY] == city][Columns.PATIENT_ID].count()
        folium.CircleMarker(
            location=coordinates,
            radius=count / 10,
            popup=f"{city}: {count} patients",
            color="blue",
            fill=True,
            fill_color="blue"
        ).add_to(m)

    m.save(os.path.join(Constants.Paths.PLOT_OUTPUT_DIR, "geographic_heatmap.html"))


# Survival Analysis (Kaplan-Meier)
def survival_analysis(dataframe: pandas.DataFrame):
    kmf = KaplanMeierFitter()
    survival_times = dataframe[Columns.LENGTH_OF_STAY]
    kmf.fit(durations=survival_times, event_observed=(survival_times > 0))

    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.title("Survival Analysis: Length of Stay")
    save_plot(plt.gcf(), "survival_analysis")


def analyze_column_statistics(dataframe: pandas.DataFrame):
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
    save_plot(plt.gcf(), "numeric_columns_distribution")
    log_time_info("Numeric Statistics:\n", numeric_stats)
    log_time_info("\nCategorical Statistics:\n", categorical_stats)
    log_time_info("\nDate Statistics:\n", date_stats)


def analyze_demographics(dataframe: pandas.DataFrame):
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
    dataframe[Columns.AGE].plot(
        kind="hist", bins=20, color="g", edgecolor="black"
    )
    plt.title("Age Distribution")
    save_plot(plt.gcf(), "age_distribution")
    log_time_info("\nDemographics Analysis:", demographics)


def analyze_disease_treatment(dataframe: pandas.DataFrame):
    disease_treatment_analysis = {
        Constants.AnalysisKeys.MOST_FREQUENT_DISEASES: dataframe[
            Columns.DISEASE_DIAGNOSED
        ]
        .value_counts()
        .head(),
    }

    # Plot most frequent diseases
    plt.figure(figsize=(10, 6))
    dataframe[Columns.DISEASE_DIAGNOSED].value_counts().head().plot(kind="bar", color="b")
    plt.title("Most Frequent Diseases")
    save_plot(plt.gcf(), "most_frequent_diseases")
    log_time_info("\nDisease and Treatment Analysis:", disease_treatment_analysis)


def analyze_resource_allocation(dataframe: pandas.DataFrame):
    hospital_volume: pandas.Series = dataframe[Columns.HOSPITAL].value_counts()

    # Plot hospital volume
    plt.figure(figsize=(10, 6))
    hospital_volume.plot(kind="bar", color="m")
    plt.title("Hospital Volume Distribution")
    save_plot(plt.gcf(), "hospital_volume_distribution")
    log_time_info("\nResource Allocation:", hospital_volume)


def analyze_funding_type(dataframe: pandas.DataFrame):
    funding_analysis: pandas.Series = dataframe[
        Constants.AnalysisKeys.CORPORATE_CLIENT___OWN_SELF
    ].value_counts()

    # Plot funding type distribution
    plt.figure(figsize=(10, 6))
    funding_analysis.plot(kind="bar", color="y")
    plt.title("Funding Type Distribution")
    save_plot(plt.gcf(), "funding_type_distribution")
    log_time_info("\nFunding Type Analysis:", funding_analysis)


# Length of Stay Analysis
def analyze_los(dataframe: pandas.DataFrame):
    # Ensure that missing Length of Stay values do not cause errors
    los_analysis: pandas.Series = (
        dataframe.groupby(Columns.DISEASE_DIAGNOSED)[Columns.LENGTH_OF_STAY]
        .mean()
        .dropna()
    )

    # Plot LOS analysis by disease
    plt.figure(figsize=(10, 6))
    los_analysis.plot(kind="bar", color="r")
    plt.title("Average Length of Stay by Disease")
    save_plot(plt.gcf(), "average_los_by_disease")
    log_time_info("\nLength of Stay Analysis:", los_analysis)


# Doctor Workload Analysis (Patient Count per Doctor)
def analyze_doctor_workload(dataframe: pandas.DataFrame):
    doctor_workload = dataframe.groupby(Columns.DOCTOR_NAME)[Columns.PATIENT_ID].count()

    # Plot doctor workload
    plt.figure(figsize=(10, 6))
    doctor_workload.plot(kind="bar", color="b")
    plt.title("Doctor Workload (Patient Count)")
    save_plot(plt.gcf(), "doctor_workload")
    log_time_info("\nDoctor Workload Analysis:", doctor_workload)


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
    x = dataframe[
        [Columns.AGE, Columns.GENDER, Columns.DISEASE_DIAGNOSED, Columns.HOSPITAL]
    ]  # Add more features if necessary
    y = dataframe[Columns.LENGTH_OF_STAY]

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Standardizing features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train Random Forest Regressor
    model = RandomForestRegressor()
    model.fit(x_train_scaled, y_train)

    # Predictions and Evaluation
    y_pred = model.predict(x_test_scaled)
    log_time_info(
        "Length of Stay Prediction Mean Absolute Error:",
        mean_absolute_error(y_test, y_pred),
    )

    # Plot actual vs predicted
    figure = plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual LoS", color="g")
    plt.plot(y_pred, label="Predicted LoS", color="r", linestyle="--")
    plt.legend()
    plt.title("Actual vs Predicted Length of Stay")
    save_plot(figure, "actual_vs_predicted_los")


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
    x = dataframe[
        [Columns.AGE, Columns.GENDER, Columns.DISEASE_DIAGNOSED, Columns.HOSPITAL]
    ]  # Add more features if necessary
    y = label_encoder.fit_transform(
        dataframe[Constants.AnalysisKeys.CORPORATE_CLIENT___OWN_SELF]
    )

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Train Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Predictions and Evaluation
    y_pred = model.predict(x_test)
    log_time_info(
        "Funding Type Prediction Classification Report:\n",
        classification_report(y_test, y_pred),
    )

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=Constants.Matplotlib.CMapColors.BLUES)
    save_plot(disp.figure_, "funding_type_confusion_matrix")


# K-Means clustering to segment patients
def analyze_patient_clustering(dataframe: pandas.DataFrame):
    # Preprocessing for clustering
    label_encoder = LabelEncoder()
    dataframe[Columns.GENDER] = label_encoder.fit_transform(dataframe[Columns.GENDER])
    dataframe[Columns.DISEASE_DIAGNOSED] = label_encoder.fit_transform(
        dataframe[Columns.DISEASE_DIAGNOSED]
    )

    # Features for clustering
    x = dataframe[
        [Columns.AGE, Columns.GENDER, Columns.DISEASE_DIAGNOSED]
    ]  # Modify with relevant features

    # Apply KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    dataframe[Constants.AnalysisKeys.CLUSTER] = kmeans.fit_predict(x)

    # Plot the clusters
    scatter_plot = plt.scatter(
        dataframe[Columns.AGE],
        dataframe[Columns.GENDER],
        c=dataframe[Constants.AnalysisKeys.CLUSTER],
        cmap=Constants.Visualization.CMAP_VIRIDIS,
    )
    plt.xlabel(Columns.AGE)
    plt.ylabel(Columns.GENDER)
    plt.title("Clustering Patients")
    save_plot(scatter_plot.figure, "patient_segmentation_clusters")


def min_max_error_evaluation(dataframe: pandas.DataFrame):
    # Step 1: Data Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(dataframe[[Columns.LENGTH_OF_STAY]].values)

    # X will contain the sequence of previous LOS values
    # y will contain the next day's LOS (predicting the next value)
    x = (
        torch.tensor(data_scaled[:-1]).float().view(-1, 1, 1)
    )  # 3D shape: [samples, timestamps, features]
    y = torch.tensor(data_scaled[1:]).float()

    # Step 2: Define the LSTM Model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, lstm_x):
            out, _ = self.lstm(lstm_x)
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
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            log_time_info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Step 6: Predicting and Inverse Scaling
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        predicted = model(x)

    # Convert predictions back to an original scale
    predicted_values = scaler.inverse_transform(predicted.numpy())
    actual_values = scaler.inverse_transform(y.numpy())

    # Step 7: Evaluate the Model
    mae = mean_absolute_error(actual_values, predicted_values)
    log_time_info(f"Mean Absolute Error: {mae:.4f}")

    # plt.plot(actual_values, label="Actual LOS")
    # plt.plot(predicted_values, label="Predicted LOS")
    # plt.legend()
    # plt.show()


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

    log_time_info("F-Statistic: ", f_stat)
    log_time_info("P-Value:", p_value)


def quick_two_way_anova_analysis(dataframe: pandas.DataFrame):
    # Fit the model
    formula_mapping = {
        "Length of Stay": "Length_of_Stay",
        "Disease Diagnosed": "Disease_Diagnosed",
        "Gender": "Gender"
    }

    renamed_dataframe = dataframe.rename(columns=formula_mapping)

    model = ols(
        "Length_of_Stay ~ C(Disease_Diagnosed) + C(Gender) + C(Disease_Diagnosed):C(Gender)",
        data=renamed_dataframe,
    ).fit()

    # Perform Two-Way ANOVA
    anova_results = sm.stats.anova_lm(model, type=2)
    log_time_info("ANOVA Results:\n", anova_results)

    # Plot ANOVA results
    plt.figure(figsize=(10, 6))
    anova_results.plot(kind="bar", color="c")
    plt.title("ANOVA Results: Treatment Impact on Length of Stay")
    save_plot(plt.gcf(), "anova_treatment_impact_los")


def quick_chi_square_analysis(dataframe: pandas.DataFrame):
    # Contingency table between Disease Diagnosed and Funding Type
    contingency_table = pandas.crosstab(
        dataframe[Columns.DISEASE_DIAGNOSED],
        dataframe[Constants.AnalysisKeys.CORPORATE_CLIENT___OWN_SELF],
    )

    # Perform Chi-Squared test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    # Display results
    log_time_info(f"Chi-Squared Test Statistic: {chi2_stat}")
    log_time_info(f"P-Value: {p_value}")
    log_time_info(f"Degrees of Freedom: {dof}")
    log_time_info(f"Expected Frequencies:\n{expected}")

    # Plot contingency table as heatmap
    plt.figure(figsize=(10, 6))
    heatmap_plot_figure = sns.heatmap(
        contingency_table, annot=True, cmap=Constants.Matplotlib.CMapColors.BLUES, fmt="d"
    )
    heatmap_plot_figure.set_title("Chi-Squared Test: Disease vs Gender Distribution")
    save_plot(plt.gcf().figure, "chi_squared_disease_gender")


def quick_mean_absolute_error_with_ridge_analysis(dataframe: pandas.DataFrame):
    # Preprocessing
    x = dataframe[[Columns.AGE, Columns.GENDER, Columns.DISEASE_DIAGNOSED, Columns.HOSPITAL]]
    y = dataframe[Columns.LENGTH_OF_STAY]

    # Train/Test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train Ridge Regression Model
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_scaled, y_train)

    # Predictions
    y_pred = ridge.predict(x_test_scaled)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    log_time_info(f"Mean Absolute Error (Ridge Regression): {mae}")

    # Plot actual vs predicted
    plot_figure = plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual LoS", color="g")
    plt.plot(y_pred, label="Predicted LoS", color="r", linestyle="--")
    plt.legend()
    plt.title("Ridge Regression: Actual vs Predicted Length of Stay")
    save_plot(plot_figure, "ridge_regression_actual_vs_predicted_los")


def quick_pca_analysis(dataframe: pandas.DataFrame):
    # Preprocessing
    x = dataframe[
        [Columns.AGE, Columns.GENDER, Columns.DISEASE_DIAGNOSED, Columns.HOSPITAL]
    ]

    # Standardizing features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Apply PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    # Visualize PCA components
    scatter_plot = plt.scatter(
        x_pca[:, 0], x_pca[:, 1], c=dataframe[Columns.DISEASE_DIAGNOSED]
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of Healthcare Data")
    save_plot(scatter_plot.figure, "pca_patient_data")


def generate_main_charts(dataframe: pandas.DataFrame) -> None:
    for chart_function in [
        plot_and_save_pie_charts,  # Pie Charts
        plot_and_save_donut_charts,  # Donut Charts
        plot_and_save_histograms,  # Histograms
        plot_and_save_bar_charts,  # Bar Charts
        plot_and_save_line_charts,  # Line Charts
        plot_and_save_heatmaps,  # Heatmaps
        plot_and_save_box_plots,  # Box Plots
        plot_and_save_scatter_plots,  # Scatter Plots
        plot_and_save_violin_plots,  # Violin Plots
        plot_and_save_bubble_charts,  # Bubble Charts
        plot_and_save_time_series_plots,  # Time Series Plots
        plot_and_save_pair_plots,  # Pair Plots
        plot_and_save_tree_maps,  # Tree Maps
        plot_and_save_word_clouds,  # Word Clouds
        plot_and_save_clustering_visualizations,  # Clustering Visualizations
        plot_and_save_radar_charts,  # Radar Charts
        # plot_and_save_geographic_visualizations,  # Geographic Visualizations
        plot_and_save_sankey_diagrams,  # Sankey Diagrams
        plot_and_save_funnel_charts,  # Funnel Charts
        plot_and_save_animated_charts_disease_trends  # Animated Charts
    ]:
        chart_function(dataframe=dataframe)


def plot_and_save_radar_charts(dataframe):
    def hospital_metrics_radar_chart():
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
        save_plot(plt.gcf().figure, "radar_chart_hospital_metrics")

    hospital_metrics_radar_chart()


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def plot_and_save_clustering_visualizations(dataframe):
    def plot_cluster_heatmap():
        dataframe_encoded = dataframe.copy()
        label_encoder = LabelEncoder()

        # Apply LabelEncoder only to categorical columns
        for col in [Columns.GENDER, Columns.DISEASE_DIAGNOSED, Columns.HOSPITAL]:
            if dataframe_encoded[col].dtype == 'object':  # Check if the column is categorical
                dataframe_encoded[col] = label_encoder.fit_transform(dataframe_encoded[col])

        # Select only numeric columns for correlation
        numeric_dataframe = dataframe_encoded.select_dtypes(include=['number'])

        # Compute the correlation matrix for numeric columns only
        cluster_corr = numeric_dataframe.corr()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        cluster_corr_plot_figure = sns.heatmap(cluster_corr, annot=True, cmap=Constants.Matplotlib.CMapColors.COOL_WARM)
        cluster_corr_plot_figure.set_title("Cluster Heatmap")

        # Save the plot
        save_plot(plt.gcf(), "cluster_heatmap")

    def plot_los_by_funding_type():
        plt.figure(figsize=(12, 8))
        corporate_client_boxplot_figure = sns.boxplot(
            x=Constants.AnalysisKeys.CORPORATE_CLIENT___OWN_SELF,
            y=Columns.LENGTH_OF_STAY,
            data=dataframe,
        )
        corporate_client_boxplot_figure.set_title("Length of Stay by Funding Type")
        save_plot(plt.gcf(), "los_by_funding_type")

    # Plot both visualizations
    for plot_function in [
        plot_cluster_heatmap,
        plot_los_by_funding_type
    ]:
        plot_function()


def plot_and_save_word_clouds(dataframe):
    def plot_word_cloud(column: str, title: str, filename: str) -> None:
        text = " ".join(dataframe[column].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text
        )
        plt.figure(figsize=(12, 8))
        imshow_figure = plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        save_plot(imshow_figure.figure, filename)

    def generate_word_clouds():
        plot_word_cloud(
            Columns.DISEASE_DIAGNOSED,
            "Common Diseases Diagnosed",
            "common_diseases_wordcloud",
        )
        plot_word_cloud(
            Columns.DOCTOR_NAME,
            "Frequent Doctor Names",
            "doctor_names_wordcloud",
        )

    generate_word_clouds()


def plot_and_save_tree_maps(dataframe):
    def plot_disease_distribution_treemap():
        disease_counts = dataframe[Columns.DISEASE_DIAGNOSED].value_counts()
        squarify_plot_figure = squarify.plot(
            sizes=disease_counts.values,
            label=disease_counts.index,
            alpha=0.8,
            color=sns.color_palette("Paired"),
        )
        squarify_plot_figure.set_title("Disease Distribution Tree Map")
        plt.axis("off")
        save_plot(plt.gcf(), "disease_distribution_treemap")

    plot_disease_distribution_treemap()


def plot_and_save_pair_plots(dataframe):
    def plot_pairwise_relationships():
        pair_plot_figure = sns.pairplot(
            dataframe,
            vars=[Columns.AGE, Columns.LENGTH_OF_STAY],
            hue=Columns.GENDER,
            palette="husl",
        )
        plt.title("Pairwise Relationships")
        save_plot(pair_plot_figure.figure, "pairwise_relationships")

    plot_pairwise_relationships()


def plot_and_save_time_series_plots(dataframe):
    def plot_daily_admissions_trends():
        daily_admissions = dataframe.groupby(dataframe[Columns.ADMIT_DATE].dt.date)[
            Columns.PATIENT_ID
        ].count()
        daily_admissions_plot_figure = daily_admissions.plot(
            kind="line", figsize=(12, 6), marker="o", color="blue"
        )
        daily_admissions_plot_figure.set_title("Daily Admissions Trends")
        save_plot(plt.gcf(), "daily_admissions_trends")

    def plot_seasonal_los_trends():
        dataframe["Season"] = dataframe[Columns.ADMIT_DATE].dt.month % 12 // 3 + 1
        seasonal_los = dataframe.groupby("Season")[Columns.LENGTH_OF_STAY].mean()
        seasonal_los_plot_figure = seasonal_los.plot(
            kind="bar", color="gold", figsize=(10, 6)
        )
        seasonal_los_plot_figure.set_title("Seasonal Length of Stay Trends")
        save_plot(plt.gcf(), "seasonal_los_trends")

    def plot_disease_trends_over_time():
        monthly_disease_trends = (
            dataframe.groupby(
                [dataframe[Columns.ADMIT_DATE].dt.to_period("M"), Columns.DISEASE_DIAGNOSED]
            )
            .size()
            .unstack()
            .fillna(0)
        )
        monthly_disease_trends_plot_figure = monthly_disease_trends.plot(
            kind="line", figsize=(14, 8), colormap="tab10"
        )
        monthly_disease_trends_plot_figure.set_title("Disease Trends Over Time")
        save_plot(plt.gcf(), "disease_trends_over_time")

    for plot_function in [
        plot_daily_admissions_trends,
        plot_seasonal_los_trends,
        plot_disease_trends_over_time
    ]:
        plot_function()


def plot_and_save_bubble_charts(dataframe):
    def hospital_vs_patients_with_length_of_stay_as_bubble_size():
        hospital_patient_los = dataframe.groupby(Columns.HOSPITAL).agg(
            patient_count=(Columns.PATIENT_ID, "count"),
            avg_los=(Columns.LENGTH_OF_STAY, "mean"),
        )
        plt.figure(figsize=(12, 8))
        scatter_plot = plt.scatter(
            hospital_patient_los.index,
            hospital_patient_los["patient_count"],
            s=hospital_patient_los["avg_los"] * 50,  # Scale bubble size
            alpha=0.6,
        )
        plt.xlabel("Hospital")
        plt.ylabel("Number of Patients")
        plt.title("Hospital vs Patients with Bubble Size as Length of Stay")
        save_plot(scatter_plot.figure, "hospital_bubble_chart")

    hospital_vs_patients_with_length_of_stay_as_bubble_size()


def plot_and_save_violin_plots(dataframe):
    def age_distribution_by_gender():
        plt.figure(figsize=(10, 6))
        violin_plot_figure = sns.violinplot(x=Columns.GENDER, y=Columns.AGE, data=dataframe)
        violin_plot_figure.set_title("Age Distribution by Gender")
        save_plot(plt.gcf(), "age_distribution_violin")

    age_distribution_by_gender()


def plot_and_save_scatter_plots(dataframe):
    def age_vs_length_of_stay_colored_by_gender():
        plt.figure(figsize=(10, 6))
        scatterplot_figure = sns.scatterplot(
            data=dataframe, x=Columns.AGE, y=Columns.LENGTH_OF_STAY, hue=Columns.GENDER
        )
        scatterplot_figure.set_title("Age vs Length of Stay by Gender")
        save_plot(plt.gcf(), "age_vs_los_scatter")

    age_vs_length_of_stay_colored_by_gender()


def plot_and_save_box_plots(dataframe):
    def length_of_stay_by_disease_diagnosed():
        plt.figure(figsize=(12, 8))
        boxplot_figure = sns.boxplot(
            x=Columns.DISEASE_DIAGNOSED, y=Columns.LENGTH_OF_STAY, data=dataframe
        )
        plt.xticks(rotation=45)
        boxplot_figure.set_title("Length of Stay Distribution by Disease Diagnosed")
        save_plot(plt.gcf(), "length_of_stay_boxplot")

    length_of_stay_by_disease_diagnosed()


def plot_and_save_heatmaps(dataframe: pandas.DataFrame):
    def correlation_between_numeric_columns():
        # Select only numeric columns
        numeric_dataframe = dataframe.select_dtypes(include=['number'])

        # Compute the correlation matrix for numeric columns only
        correlation_matrix = numeric_dataframe.corr()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        heatmap_plot_figure = sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f"
        )
        heatmap_plot_figure.set_title("Correlation Heatmap of Numeric Columns")

        # Save the plot
        save_plot(plt.gcf(), "correlation_heatmap")

    correlation_between_numeric_columns()


def plot_and_save_line_charts(dataframe):
    def monthly_admissions_trends():
        dataframe[Columns.ADMIT_DATE] = pandas.to_datetime(
            dataframe[Columns.ADMIT_DATE], errors="coerce"
        )
        monthly_admissions = (
            dataframe.set_index(Columns.ADMIT_DATE)
            .resample("ME")[Columns.PATIENT_ID]
            .count()
        )
        line_plot = monthly_admissions.plot(
            kind="line", marker="o", figsize=(10, 6), color="green"
        )
        line_plot.set_title("Monthly Admission Trends")
        save_plot(plt.gcf(), "monthly_admissions_trends")

    monthly_admissions_trends()


def plot_and_save_bar_charts(dataframe):
    def average_length_of_stay_per_disease_diagnosed():
        avg_los_disease = dataframe.groupby(Columns.DISEASE_DIAGNOSED)[
            Columns.LENGTH_OF_STAY
        ].mean()
        avg_loss_bar_plot = avg_los_disease.plot(
            kind="bar", color=Constants.Matplotlib.Colors.SKY_BLUE, figsize=(10, 6)
        )
        avg_loss_bar_plot.set_title("Average Length of Stay per Disease Diagnosed")
        save_plot(plt.gcf(), "avg_los_disease")

    def plot_top_diseases_by_city():
        diseases_by_city = (
            dataframe.groupby([Columns.CITY, Columns.DISEASE_DIAGNOSED])[Columns.PATIENT_ID]
            .count()
            .unstack()
            .fillna(0)
        )
        diseases_by_city_plot_figure = diseases_by_city.plot(
            kind="bar", stacked=True, figsize=(12, 8), colormap="tab20"
        )
        diseases_by_city_plot_figure.set_title("Top Diseases by City")
        plt.ylabel("Number of Patients")
        plt.xticks(rotation=45)
        save_plot(plt.gcf(), "top_diseases_by_city")

    def plot_avg_patient_age_by_hospital():
        avg_age_hospital = dataframe.groupby(Columns.HOSPITAL)[Columns.AGE].mean()
        hospital_plot_figure = avg_age_hospital.plot(
            kind="bar", color="orchid", figsize=(10, 6)
        )
        hospital_plot_figure.set_title("Average Patient Age by Hospital")
        save_plot(plt.gcf(), "avg_patient_age_hospital")

        # plot_doctor_patient_ratio
        doctor_patient_counts = dataframe.groupby(Columns.DOCTOR_NAME)[
            Columns.PATIENT_ID
        ].count()
        counts_plot_figure = doctor_patient_counts.plot(
            kind="bar", figsize=(12, 6), color="teal"
        )
        counts_plot_figure.set_title("Doctor-Patient Ratio")
        save_plot(plt.gcf(), "doctor_patient_ratio")

    for plot_function in [
        average_length_of_stay_per_disease_diagnosed,
        plot_top_diseases_by_city,
        plot_avg_patient_age_by_hospital
    ]:
        plot_function()


def plot_and_save_histograms(dataframe):
    def distribution_of_age_with_gender_segmentation():
        plt.figure(figsize=(10, 6))
        hist_plot_figure = sns.histplot(
            data=dataframe, x=Columns.AGE, hue=Columns.GENDER, kde=True, bins=20
        )
        hist_plot_figure.set_title("Age Distribution by Gender")
        save_plot(plt.gcf(), "age_distribution_gender")

    distribution_of_age_with_gender_segmentation()


def plot_and_save_donut_charts(dataframe):
    def breakdown_of_disease_diagnosed_by_gender():
        disease_gender = dataframe.groupby(Columns.GENDER)[
            Columns.DISEASE_DIAGNOSED
        ].value_counts()
        disease_gender.plot.pie(autopct="%1.1f%%", figsize=(8, 8))
        circle = plt.Circle((0, 0), 0.7, color="white")
        plt.gca().add_artist(circle)
        plt.title("Disease Diagnosed by Gender")
        save_plot(circle.figure, "disease_gender_donut")

    breakdown_of_disease_diagnosed_by_gender()


def plot_and_save_pie_charts(dataframe):
    def distribution_of_gender_per_city():
        gender_city = (
            dataframe.groupby(Columns.CITY)[Columns.GENDER].value_counts().unstack()
        )
        fig, axes = plt.subplots(1, len(gender_city.columns), figsize=(15, 6))
        for i, column in enumerate(gender_city.columns):
            gender_city[column].plot(kind="pie", ax=axes[i], autopct="%1.1f%%", startangle=90, legend=False)
            axes[i].set_ylabel('')  # Hide the y-axis label for clarity
            axes[i].set_title(f"{column} Distribution")
        plt.tight_layout()  # Adjust layout for better fit
        save_plot(fig, "gender_distribution_city")

    distribution_of_gender_per_city()


def plot_and_save_geographic_visualizations(dataframe):
    def plot_admissions_by_region():
        # Geolocate cities to get latitude and longitude
        geolocator = Nominatim(user_agent="geoapi")
        city_coordinates = {}
        for city in dataframe[Columns.CITY].unique():
            location = geolocator.geocode(city)
            if location:
                city_coordinates[city] = (location.latitude, location.longitude)

        # Prepare a map
        admission_map = folium.Map(
            location=[20, 78], zoom_start=5
        )  # Center map over an example region (India here)

        # Add markers for cities
        for city, coordinate in city_coordinates.items():
            patient_count = dataframe[dataframe[Columns.CITY] == city][
                Columns.PATIENT_ID
            ].count()
            folium.CircleMarker(
                location=coordinate,
                radius=patient_count / 10,  # Scale the radius
                popup=f"{city}: {patient_count} patients",
                color="blue",
                fill=True,
                fill_color="blue",
            ).add_to(admission_map)

        # Save the map as HTML
        save_plot(plt.gcf(), "admissions_by_region.html")

    plot_admissions_by_region()


def plot_and_save_sankey_diagrams(dataframe):
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


def plot_and_save_funnel_charts(dataframe):
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


def plot_and_save_animated_charts_disease_trends(dataframe):
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
    dataframe = initialize()
    # Generate main charts
    generate_main_charts(dataframe=dataframe)

    # Basic data analysis
    analyze_column_statistics(dataframe=dataframe)
    analyze_demographics(dataframe=dataframe)
    analyze_disease_treatment(dataframe=dataframe)
    analyze_resource_allocation(dataframe=dataframe)
    analyze_funding_type(dataframe=dataframe)
    analyze_doctor_workload(dataframe=dataframe)
    analyze_los(dataframe=dataframe)

    # ML analysis
    analyze_los_prediction(dataframe=dataframe)
    analyze_funding_type_prediction(dataframe=dataframe)
    analyze_patient_clustering(dataframe=dataframe)
    min_max_error_evaluation(dataframe=dataframe)

    # Statistical analysis (ANOVA, etc.)
    quick_anova_analysis(dataframe=dataframe)
    quick_two_way_anova_analysis(dataframe=dataframe)
    quick_chi_square_analysis(dataframe=dataframe)
    quick_mean_absolute_error_with_ridge_analysis(dataframe=dataframe)
    quick_pca_analysis(dataframe=dataframe)

    # Some advanced analysis
    dataframe = clean_data(dataframe)
    enhanced_correlation_heatmap(dataframe)
    enhanced_los_prediction(dataframe)
    # geographic_heatmap(dataframe)
    survival_analysis(dataframe)


def initialize():
    file_path = "data/HealthcareData.csv"
    dataframe: pandas.DataFrame = pandas.read_csv(file_path)

    # Date columns - time range statistics
    dataframe[Columns.ADMIT_DATE] = pandas.to_datetime(
        dataframe[Columns.ADMIT_DATE], errors="coerce"
    )
    dataframe[Columns.DISCHARGE_DATE] = pandas.to_datetime(
        dataframe[Columns.DISCHARGE_DATE], errors="coerce"
    )

    # Convert the 'DISCHARGE_DATE' and 'ADMIT_DATE' columns to datetime
    dataframe[Columns.DISCHARGE_DATE] = pandas.to_datetime(dataframe[Columns.DISCHARGE_DATE], format='%Y-%m-%d')
    dataframe[Columns.ADMIT_DATE] = pandas.to_datetime(dataframe[Columns.ADMIT_DATE], format='%Y-%m-%d')

    # Calculate the 'LENGTH_OF_STAY' by subtracting the dates
    dataframe[Columns.LENGTH_OF_STAY] = (dataframe[Columns.DISCHARGE_DATE] - dataframe[Columns.ADMIT_DATE]).dt.days

    # Get all column names
    log_time_info(f"Columns: {dataframe.columns}")

    return dataframe


def log_time_info(*message_args):
    """
    Helper function to log a message with the current timestamp.
    Accepts multiple arguments similar to print and logs them as a single string.
    """
    message = " ".join(map(str, message_args))  # Convert all arguments to strings and join with a space
    logger.info(f"[{datetime.now()}] {message}")


def initialize_logger():
    global logger

    # Initialize the logger only once
    if logger is not None:
        return

    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the minimum logging level (can be adjusted to DEBUG, INFO, etc.)
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs to a file
    file_handler = logging.FileHandler('logs/main.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler that logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a log formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Add the formatter to both handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def chatgpt_feedback(log_text: str) -> str:
    """
    Send the ERD text to ChatGPT and retrieve the feedback in the requested format.

    Args:
        log_text (str): The ERD text to send to ChatGPT.

    Returns:
        str: The improved ERD text from ChatGPT.
    """

    global OPENAI_CLIENT
    system_content = (
        "You are an AI assistant that specializes in analyzing and generating reports from logs. "
        "Your task is to review the log entries and derive key insights such as trends, anomalies, and important observations. "
        "You should identify patterns in data, such as distributions, correlations, and outliers, and summarize them clearly. "
        "Additionally, you should generate a well-structured, concise report highlighting key takeaways and relevant statistics. "
        "Return only the generated report, with no preamble or explanation."
    )

    user_content = (
        f"Please analyze the following log data and generate a detailed report summarizing the insights. "
        f"Focus on trends in the numerical data, correlations between variables, anomalies in the categorical data, and any "
        f"interesting patterns or observations. Provide a well-organized report with sections for each major insight:\n\n{log_text}"
    )

    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
        )

        # Extract the generated content
        feedback = response.choices[0].message.content.strip()

        # Safeguard: Verify if the response contains valid ERD-like structure
        if not feedback:
            raise ValueError("ChatGPT returned an empty response. No ERD output found.")

        # Log successful interaction
        log_time_info("Received feedback from ChatGPT.")
        return feedback.replace("```", "")

    except openai.AuthenticationError:
        log_time_info("Authentication error: Ensure the OPENAI_API_KEY is set correctly.")
        raise

    except openai.APIConnectionError:
        log_time_info("Network issue: Unable to connect to the OpenAI API.")
        raise

    except openai.OpenAIError as e:
        log_time_info(f"OpenAI API error: {e}")
        raise

    except Exception as e:
        log_time_info(f"Unexpected error while interacting with ChatGPT: {e}")
        raise


def generate_gpt_insights():
    try:
        with open(f"{Constants.Paths.LOGS_DIR}/main.log", "r") as f:
            main_log = f.read()
    except FileNotFoundError:
        logger.error("main.log file not found.")
        return

    feedback = chatgpt_feedback(main_log)
    if not feedback:
        logger.error("No feedback generated.")
        return

    try:
        i = 0
        while os.path.exists(f"{Constants.Paths.LOGS_DIR}/feedback_{i}.log"):
            i += 1
        with open(f"{Constants.Paths.LOGS_DIR}/feedback_{i}.log", "w") as f:
            f.write(feedback)
    except FileNotFoundError:
        logger.error("feedback.log file could not be saved to.")
        return


if __name__ == "__main__":
    set_env_vars()
    for folder in [Constants.Paths.LOGS_DIR, Constants.Paths.PLOT_OUTPUT_DIR]:
        create_output_folder(folder)

    initialize_logger()

    parser = argparse.ArgumentParser(description="Run healthcare data analysis or generate GPT insights.")
    parser.add_argument("--mode", choices=["analyze", "insights"], required=True,
                        help="Mode to run: 'analyze' for data analysis, 'insights' for GPT insights.")
    args = parser.parse_args()

    if args.mode == "analyze":
        main()
    elif args.mode == "insights":
        generate_gpt_insights()
