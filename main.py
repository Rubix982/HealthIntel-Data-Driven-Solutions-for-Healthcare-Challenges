import pandas
from typing import Union


def analyze_column_statistics(
    dataframe: pandas.DataFrame,
) -> Union[pandas.Series, pandas.Series, dict[str, tuple]]:

    # Numeric columns - descriptive statistics
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns
    numeric_stats: pandas.Series = dataframe[numeric_columns].describe()

    # Categorical columns - frequency counts
    categorical_columns = dataframe.select_dtypes(include=["object"]).columns
    categorical_stats: pandas.Series = dataframe[categorical_columns].describe()

    # Date columns - time range statistics
    dataframe["Admit Date"] = pandas.to_datetime(
        dataframe["Admit Date"], errors="coerce"
    )
    dataframe["Discharge Date"] = pandas.to_datetime(
        dataframe["Discharge Date"], errors="coerce"
    )
    data_columns = dataframe.select_dtypes(include=["datatime"]).columns
    date_stats: dict[str, tuple] = {
        col: (dataframe[col].min(), dataframe[col].max()) for col in data_columns
    }

    return numeric_stats, categorical_stats, date_stats


def analyze_demographics(dataframe: pandas.DataFrame) -> dict:
    demographics: dict = {
        "Age Distribution": dataframe["Age"].describe(),
        "Gender Distribution": dataframe["Gender"].value_counts(),
        "Geography Distribution": dataframe["City"].value_counts(),
    }
    return demographics


def analyze_disease_treatment(dataframe: pandas.DataFrame) -> dict:
    disease_treatment_analysis = {
        "Most Frequent Diseases": dataframe["Disease Diagnosed"].value_counts().head(),
    }
    return disease_treatment_analysis


def analyze_resource_allocation(dataframe: pandas.DataFrame) -> pandas.Series[int]:
    hospital_volume: pandas.Series[int] = dataframe["Hospital"].value_counts()
    return hospital_volume


def analyze_funding_type(dataframe: pandas.DataFrame) -> pandas.Series[int]:
    funding_analysis: pandas.Series[int] = dataframe[
        "Corporate Client / Own Self"
    ].value_counts()
    return funding_analysis


# Length of Stay Analysis
def analyze_los(dataframe: pandas.DataFrame) -> pandas.Series[any]:
    los_analysis: pandas.Series[any] = dataframe.groupby("Disease Diagnosed")[
        "Length Of Stay"
    ].mean()
    return los_analysis


# Doctor Workload Analysis (Patient Count per Doctor)
def analyze_doctor_workload(dataframe: pandas.DataFrame):
    doctor_workload = dataframe.groupby("Doctor Name")["Patient ID"].count()
    return doctor_workload


def main():
    file_path = "data/HealthcareData.csv"
    dataframe: pandas.DataFrame = pandas.read_csv(file_path)

    # Get all column names
    column_names = dataframe.columns
    print(f"Columns: {column_names}")

    dataframe["Length of Stay"] = (
        dataframe["Discharge Date"] - dataframe["Admit Date"]
    ).dt.days

    numeric_stats, categorical_stats, date_stats = analyze_column_statistics(dataframe)
    print("Numeric Statistics:\n", numeric_stats)
    print("\nCategorical Statistics:\n", categorical_stats)
    print("\nDate Statistics:\n", date_stats)

    demographics = analyze_demographics(dataframe=dataframe)
    print("Demographics Analysis:", demographics)

    disease_treatment = analyze_disease_treatment(dataframe=dataframe)
    print("Disease and Treatment Analysis:", disease_treatment)

    resource_allocation = analyze_resource_allocation(dataframe=dataframe)
    print("Resource Allocation:", resource_allocation)

    funding_type = analyze_funding_type(dataframe=dataframe)
    print("Funding Type Analysis:", funding_type)

    doctor_workload = analyze_doctor_workload(dataframe=dataframe)
    print("Doctor Workload Analysis:", doctor_workload)

    los_analysis = analyze_los(dataframe=dataframe)
    print("Length of Stay Analysis:", los_analysis)


if __name__ == "__main__":
    main()
