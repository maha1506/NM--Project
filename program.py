# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set visualization style
sns.set(style="whitegrid")

# Load dataset
def load_data(file_path):
    
    Load employee performance data from a CSV file.
    
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

# Inspect dataset
def inspect_data(df):
    
    Print basic info and preview of the dataset.
    
    print("\n--- Data Preview (Head 10 Rows) ---")
    print(df.head(10))
    print("\n--- Data Summary ---")
    print(df.info())
    print("\n--- Data Description ---")
    print(df.describe(include='all'))

# Handle missing values
def clean_data(df):
    
  Handle missing values with filling or dropping.
    
    print("\n--- Missing Values Before Clean ---")
    print(df.isnull().sum())

    # For simplicity, forward fill missing data
    df.fillna(method='ffill', inplace=True)

    # If any missing remain, drop them
    df.dropna(inplace=True)

    print("\n--- Missing Values After Clean ---")
    print(df.isnull().sum())
    return df

# Convert date column to datetime
def convert_dates(df, date_col='date'):
    
    Convert the date column to datetime type.
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    # Drop rows where date conversion failed
    df = df.dropna(subset=[date_col])
    return df

# Add calculated fields
def add_kpis(df):
    
    Add important Key Performance Indicators (KPIs) to the dataframe.
    
    # Productivity Score = tasks completed divided by hours worked
    df['productivity_score'] = df['tasks_completed'] / df['hours_worked']

    # Attendance rate as ratio of days present over total working days
    if 'days_present' in df.columns and 'total_working_days' in df.columns:
        df['attendance_rate'] = df['days_present'] / df['total_working_days']
    else:
        df['attendance_rate'] = np.nan

    # Calculate efficiency: tasks completed per day present (if data available)
    if 'days_present' in df.columns:
        df['efficiency'] = df['tasks_completed'] / df['days_present']
    else:
        df['efficiency'] = np.nan

    # Replace inf or NaN in productivity_score with 0
    df['productivity_score'].replace([np.inf, -np.inf], 0, inplace=True)
    df['productivity_score'].fillna(0, inplace=True)

    return df

# Plot overall productivity trends over time
def plot_productivity_trends(df):
    
    Plot average productivity score trends over time.
    
    trend = df.groupby('date')['productivity_score'].mean().reset_index()

    plt.figure(figsize=(14, 7))
    sns.lineplot(x='date', y='productivity_score', data=trend, marker='o')
    plt.title('Average Employee Productivity Trends Over Time', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Average Productivity Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot attendance trends over time (if attendance data available)
def plot_attendance_trends(df):
    
    Visualize average attendance rate over time.
        if df['attendance_rate'].isnull().all():
        print("Attendance rate data not available, skipping attendance trend plot.")
        return

    trend = df.groupby('date')['attendance_rate'].mean().reset_index()

    plt.figure(figsize=(14, 7))
    sns.lineplot(x='date', y='attendance_rate', data=trend, marker='o', color='green')
    plt.title('Average Employee Attendance Rate Over Time', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Average Attendance Rate')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Distribution of productivity scores
def plot_productivity_distribution(df):
    
    Plot histogram and boxplot of employee productivity scores.
    
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df['productivity_score'], bins=30, kde=True, color='blue')
    plt.title('Productivity Score Distribution')
    plt.xlabel('Productivity Score')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df['productivity_score'], color='cyan')
    plt.title('Productivity Score Boxplot')
    plt.xlabel('Productivity Score')

    plt.tight_layout()
    plt.show()

# Correlation heatmap for performance-related columns
def plot_correlation_heatmap(df):
    
    Plot a correlation heatmap of numeric performance metrics.
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Employee Performance Metrics', fontsize=16)
    plt.show()

# Top performers identification
def identify_top_performers(df, kpi_col='productivity_score', top_n=10):
    
    Identify top N performers based on the KPI column.
    
    top_performers = df.groupby('employee_id')[kpi_col].mean().sort_values(ascending=False).head(top_n)
    print(f"\nTop {top_n} Performers by Average {kpi_col}:\n")
    print(top_performers)

# Productivity by department or team if available
def productivity_by_group(df, group_col='department'):
    
    Visualize productivity scores grouped by specified column.
    
    if group_col not in df.columns:
        print(f"Column '{group_col}' not found in dataset. Skipping group productivity plot.")
        return

    plt.figure(figsize=(14, 7))
    sns.boxplot(x=group_col, y='productivity_score', data=df)
    plt.title(f'Productivity Score Distribution by {group_col.capitalize()}', fontsize=16)
    plt.xlabel(group_col.capitalize())
    plt.ylabel('Productivity Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Productivity trends for individual employees
def plot_individual_trends(df, employee_ids):
    
    Plot productivity trends over time for a list of employees.
    
    plt.figure(figsize=(14, 8))
    for emp in employee_ids:
        emp_data = df[df['employee_id'] == emp].groupby('date')['productivity_score'].mean().reset_index()
        plt.plot(emp_data['date'], emp_data['productivity_score'], marker='o', label=f'Employee {emp}')

    plt.title('Individual Employee Productivity Trends', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Productivity Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Summary statistics
def performance_summary(df):
   
    Display descriptive stats on the performance KPIs.
    
    print("\n--- Summary Statistics for KPIs ---")
    kpi_cols = ['productivity_score', 'attendance_rate', 'efficiency']
    print(df[kpi_cols].describe())

# Explore relationships through scatter plots
def scatter_plot_relationship(df, x_col, y_col):
    
    Plot scatter plot with regression line between two KPIs.
   
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Columns '{x_col}' or '{y_col}' not found in dataset.")
        return

    plt.figure(figsize=(10, 6))
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title(f'Relationship Between {x_col.capitalize()} and {y_col.capitalize()}')
    plt.xlabel(x_col.capitalize())
    plt.ylabel(y_col.capitalize())
    plt.tight_layout()
    plt.show()

# Function for detailed employee-level report for a single employee
def employee_report(df, employee_id):
    
    Provide a detailed report on performance metrics for an employee.
    
    emp_data = df[df['employee_id'] == employee_id].sort_values('date')
    if emp_data.empty:
        print(f"No data found for employee ID: {employee_id}")
        return

    print(f"\n--- Performance Report for Employee {employee_id} ---")
    print(emp_data[['date', 'tasks_completed', 'hours_worked', 'productivity_score', 'attendance_rate', 'efficiency']])

    # Plot productivity trend for this employee
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='date', y='productivity_score', data=emp_data, marker='o')
    plt.title(f'Productivity Trend for Employee {employee_id}')
    plt.xlabel('Date')
    plt.ylabel('Productivity Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main function to run all analysis
def main():
    # Path to dataset
    data_file = "employee_performance.csv"  # Change if necessary

    # Load data
    df = load_data(data_file)
    if df is None:
        return

    # Inspect
    inspect_data(df)

    # Clean & preprocess
    df = clean_data(df)
    df = convert_dates(df, date_col='date')
    df = add_kpis(df)

    # Summary stats of KPIs
    performance_summary(df)

    # Visualizations
    plot_productivity_distribution(df)
    plot_productivity_trends(df)
    plot_attendance_trends(df)
    plot_correlation_heatmap(df)

    # Top performers
    identify_top_performers(df, kpi_col='productivity_score', top_n=10)

    # Productivity by group (e.g., department or team)
    productivity_by_group(df, group_col='department')

    # Individual trends for selected employees (example employee IDs 101, 102, 103)
    emp_list = df['employee_id'].dropna().unique()[:3]  # take first 3 unique employee IDs
    plot_individual_trends(df, emp_list)

    # Example of scatter plots to examine KPI relationships
    scatter_plot_relationship(df, 'hours_worked', 'tasks_completed')
    scatter_plot_relationship(df, 'attendance_rate', 'productivity_score')

    # Detailed report for a specific employee (example employee_id=101)
    employee_report(df, employee_id=emp_list[0])

if _name_ == "_main_":
    main()
