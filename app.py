import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from xgboost import XGBRegressor

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide"
)

st.title("Sales Forecasting Dashboard")

@st.cache_data
def load_data():

    # Read CSV file
    df = pd.read_csv("Dataset.csv")

    # Convert date columns to datetime format
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    # Create Year column
    df["Year"] = df["Order Date"].dt.year

    return df


# Load dataset
df = load_data()

model = XGBRegressor()
model.load_model("xgboost_model.json")
metrics = joblib.load("metrics.pkl")

# Sidebar Navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

# Sales Overview Page
if page == "Sales Overview":

    st.header("Sales Overview Dashboard")

    # Total Sales by year
    yearly_sales = (
        df.groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8,4))
    colors = '#2DA8A8'
    ax.bar(
        yearly_sales["Year"].astype(str),
        yearly_sales["Sales"],
        color=colors
    )

    ax.set_title("Total Sales by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

    # Monthly Sales Trend
    monthly_sales = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        monthly_sales["Order Date"],
        monthly_sales["Sales"]
    )

    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    selected_region = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    selected_category = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    filtered_df = df[
        (df["Region"] == selected_region) &
        (df["Category"] == selected_category)
    ]

    st.subheader("Filtered Sales Data")

    st.dataframe(
        filtered_df[
            [
                "Order Date",
                "Region",
                "Category",
                "Sales"
            ]
        ]
    )

    filtered_sales = (
        filtered_df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        filtered_sales["Order Date"],
        filtered_sales["Sales"],
        marker="o"
    )

    ax.set_title("Filtered Monthly Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

# Forecast Explorer Page
if page == "Forecast Explorer":

    st.header("Forecast Explorer")

    selected_category = st.selectbox(
        "Select Category",
        sorted(df["Category"].unique()),
        key="forecast_category"
    )

    selected_region = st.selectbox(
        "Select Region",
        sorted(df["Region"].unique()),
        key="forecast_region"
    )

    forecast_horizon = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

# prepare data for forecasting

    filtered_df = df[
        (df["Category"] == selected_category) &
        (df["Region"] == selected_region)
    ]

    monthly = (
        filtered_df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    monthly["Month"] = monthly["Order Date"].dt.month
    monthly["Quarter"] = monthly["Order Date"].dt.quarter


    def get_season(month):

        if month in [12,1,2]:
            return 0
        elif month in [3,4,5]:
            return 1
        elif month in [6,7,8]:
            return 2
        else:
            return 3


    monthly["Season"] = monthly["Month"].apply(get_season)

    monthly["Lag_1"] = monthly["Sales"].shift(1)
    monthly["Lag_2"] = monthly["Sales"].shift(2)
    monthly["Lag_3"] = monthly["Sales"].shift(3)

    monthly["Rolling_Mean_3"] = (
        monthly["Sales"]
        .rolling(3)
        .mean()
    )

    monthly.dropna(inplace=True)

# Forecast Future Months

    future = monthly.copy()

    forecasts = []

    for _ in range(forecast_horizon):

        lag1 = future.iloc[-1]["Sales"]
        lag2 = future.iloc[-2]["Sales"]
        lag3 = future.iloc[-3]["Sales"]

        rolling = np.mean([lag1, lag2, lag3])

        next_date = (
            future.iloc[-1]["Order Date"] +
            pd.offsets.MonthEnd(1)
        )

        month = next_date.month
        quarter = next_date.quarter
        season = get_season(month)

        X_future = pd.DataFrame({

            "Lag_1":[lag1],
            "Lag_2":[lag2],
            "Lag_3":[lag3],
            "Rolling_Mean_3":[rolling],
            "Month":[month],
            "Quarter":[quarter],
            "Season":[season]

        })

        prediction = model.predict(X_future)[0]

        forecasts.append(prediction)

        new_row = pd.DataFrame({

            "Order Date":[next_date],
            "Sales":[prediction]

        })

        future = pd.concat(
            [future, new_row],
            ignore_index=True
        )
# Display Forecast Result

    forecast_table = pd.DataFrame({

        "Forecast Month": pd.date_range(

            start=monthly["Order Date"].max() + pd.offsets.MonthEnd(1),
            periods=forecast_horizon,
            freq="ME"

        ),

        "Forecast Sales": forecasts

    })

    st.subheader("Forecast")

    st.dataframe(
        forecast_table.round(2)
    )

# Display Performance Metrics

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "MAE",
            round(metrics["MAE"],2)
        )

    with col2:
        st.metric(
            "RMSE",
            round(metrics["RMSE"],2)
        )
    
# Anomaly Report Page
if page == "Anomaly Report":

    st.header("Anomaly Report")

    # Create weekly sales data
    weekly_sales = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="W"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    # Isolation Forest Model
    from sklearn.ensemble import IsolationForest

    iso_model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Anomaly"] = iso_model.fit_predict(
        weekly_sales[["Sales"]]
    )

    # Plot weekly sales
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        weekly_sales["Order Date"],
        weekly_sales["Sales"],
        label="Weekly Sales"
    )

    anomaly_points = weekly_sales[
        weekly_sales["Anomaly"] == -1
    ]

    ax.scatter(
        anomaly_points["Order Date"],
        anomaly_points["Sales"],
        color="red",
        label="Anomaly",
        s=60
    )

    ax.set_title("Isolation Forest Anomaly Detection")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    ax.legend()

    st.pyplot(fig)

    # Display Anamoly Dates
    st.subheader("Detected Anomalies")

    st.dataframe(
        anomaly_points[
            ["Order Date", "Sales"]
        ].reset_index(drop=True)
    )

# Product Demand Segments Page
if page == "Product Demand Segments":

    st.header("Product Demand Segments")

    # Aggregate Sales at Sub-Category Level
    subcat_sales = (
        df.groupby(["Sub-Category", "Year"])["Sales"].sum()
        .reset_index()
    )

    subcat_sales["Growth Rate"] = (
        subcat_sales
        .groupby("Sub-Category")["Sales"]
        .pct_change()
    )

    # Sales Volatility
    monthly_subcat = (
        df.groupby(
            [
                "Sub-Category",
                pd.Grouper(key="Order Date", freq="ME")
            ]
        )["Sales"]
        .sum()
        .reset_index()
    )

    volatility = (
        monthly_subcat
        .groupby("Sub-Category")["Sales"]
        .std()
        .reset_index(name="Sales Volatility")
    )

    # Calculate Total Sales and Average Order Value
    total_sales = (
        df.groupby("Sub-Category")["Sales"].sum()
        .reset_index(name="Total Sales Volume")
    )

    avg_order = (
        df.groupby("Sub-Category")["Sales"].mean()
        .reset_index(name="Average Order Value")
    )

    growth = (
        subcat_sales
        .groupby("Sub-Category")["Growth Rate"].mean()
        .reset_index()
    )

    # Merge all Features
    cluster_data = total_sales.merge(
        growth,
        on="Sub-Category"
    )

    cluster_data = cluster_data.merge(
        volatility,
        on="Sub-Category"
    )

    cluster_data = cluster_data.merge(
        avg_order,
        on="Sub-Category"
    )

    cluster_data.fillna(0, inplace=True)

    # Feature Scalling
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        cluster_data.drop(columns="Sub-Category")
    )

    # Apply KMeans Clustering
    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=6,
        random_state=42
    )

    cluster_data["Cluster"] = kmeans.fit_predict(X_scaled)

    # PCA for 2D Visualization
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(X_scaled)

    cluster_data["PC1"] = pca_result[:,0]
    cluster_data["PC2"] = pca_result[:,1]

    # Plot Clusters
    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        c=cluster_data["Cluster"],
        s=120
    )

    ax.set_title("Product Demand Segments")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster"
    )

    ax.add_artist(legend)
    st.pyplot(fig)

    # Assign Cluster Labels
    cluster_labels = {
    0: "High Volume, Stable Demand",
    1: "Low Volume, High Volatility",
    2: "Growing Demand",
    3: "Declining Demand",
    4: "Seasonal Products",
    5: "Moderate Demand"
}

    cluster_data["Segment"] = (
        cluster_data["Cluster"]
        .map(cluster_labels)
    )

    # Display Sub-Categorical Segment
    st.subheader("Sub-Category Demand Segments")

    st.dataframe(
        cluster_data[
            [
                "Sub-Category",
                "Cluster",
                "Segment"
            ]
        ]
    )