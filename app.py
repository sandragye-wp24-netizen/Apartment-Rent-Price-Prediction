# ============================================================
# APARTMENT RENTAL PRICE PREDICTION SYSTEM
# Streamlit Deployment Prototype
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Apartment Rental Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# 2. LOAD PREPARED DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("apartments_prepared.csv")


df = load_data()


# ============================================================
# 3. LOAD DEPLOYMENT RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("random_forest_rent_model.pkl")


deployment_model = load_model()


# ============================================================
# 4. MODEL FEATURES
# ============================================================

features = [
    "bathrooms",
    "bedrooms",
    "square_feet",
    "latitude",
    "longitude",
    "allows_cats",
    "allows_dogs",
    "has_pool",
    "has_parking",
    "has_fee",
    "has_photo",
    "state"
]

target = "price"


# ============================================================
# 5. LATEST MODEL EVALUATION RESULTS
# ============================================================

performance_df = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],

    "Training MAE": [
        369.01,
        188.75,
        192.65,
        254.37
    ],

    "Testing MAE": [
        367.70,
        235.97,
        245.24,
        259.31
    ],

    "Training RMSE": [
        599.08,
        328.30,
        285.85,
        396.23
    ],

    "Testing RMSE": [
        590.86,
        438.24,
        412.32,
        428.81
    ],

    "Training R²": [
        0.4816,
        0.8443,
        0.8820,
        0.7732
    ],

    "Testing R²": [
        0.4899,
        0.7194,
        0.7516,
        0.7313
    ]
})


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.title("🏠 Apartment Rental")

st.sidebar.write(
    "Apartment Rental Price Prediction System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Rental Price Prediction",
        "Project Overview",
        "Data Preparation",
        "Data Exploration",
        "Model Evaluation",
        "Best Model Selection"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning Deployment Prototype"
)


# ============================================================
# PAGE 1 — RENTAL PRICE PREDICTION
# ============================================================

if page == "Rental Price Prediction":

    st.title("🔮 Apartment Rental Price Prediction")

    # --------------------------------------------------------
    # USER GUIDELINE
    # --------------------------------------------------------

    st.info(
        """
        **How to Use the Predictor**

        1. Enter the apartment characteristics below.
        2. Select the relevant facilities and state.
        3. Click **Predict Rental Price**.
        4. The system will display the estimated monthly rental price.
        """
    )

    st.write(
        """
        The prediction is generated using the deployed
        Random Forest Regressor.
        """
    )

    st.divider()

    st.subheader("Apartment Information")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            help="Number of bathrooms in the apartment."
        )

        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Number of bedrooms in the apartment."
        )

        square_feet = st.number_input(
            "Apartment Size (Square Feet)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=50,
            help="Total apartment floor area in square feet."
        )

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=40.7128,
            format="%.6f",
            help="North-south geographical position."
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-74.0060,
            format="%.6f",
            help="East-west geographical position."
        )

        state_list = sorted(
            df["state"]
            .dropna()
            .astype(str)
            .unique()
        )

        state = st.selectbox(
            "State",
            state_list,
            help="State where the apartment is located."
        )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        has_photo_option = st.selectbox(
            "Listing Has Photo",
            ["Yes", "No"]
        )

        has_photo = (
            1 if has_photo_option == "Yes" else 0
        )

        allows_cats_option = st.selectbox(
            "Cats Allowed",
            ["Yes", "No"]
        )

        allows_cats = (
            1 if allows_cats_option == "Yes" else 0
        )

        allows_dogs_option = st.selectbox(
            "Dogs Allowed",
            ["Yes", "No"]
        )

        allows_dogs = (
            1 if allows_dogs_option == "Yes" else 0
        )

        has_pool_option = st.selectbox(
            "Swimming Pool Available",
            ["Yes", "No"]
        )

        has_pool = (
            1 if has_pool_option == "Yes" else 0
        )

        has_parking_option = st.selectbox(
            "Parking Available",
            ["Yes", "No"]
        )

        has_parking = (
            1 if has_parking_option == "Yes" else 0
        )

        has_fee_option = st.selectbox(
            "Rental / Application Fee",
            ["Yes", "No"]
        )

        has_fee = (
            1 if has_fee_option == "Yes" else 0
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "Predict Rental Price",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "bathrooms": [bathrooms],
            "bedrooms": [bedrooms],
            "square_feet": [square_feet],
            "latitude": [latitude],
            "longitude": [longitude],
            "allows_cats": [allows_cats],
            "allows_dogs": [allows_dogs],
            "has_pool": [has_pool],
            "has_parking": [has_parking],
            "has_fee": [has_fee],
            "has_photo": [has_photo],
            "state": [state]
        })

        try:

            prediction = deployment_model.predict(
                input_data
            )

            predicted_price = prediction[0]

            st.success(
                "Prediction completed successfully."
            )

            st.metric(
                "Estimated Monthly Rental Price",
                f"${predicted_price:,.2f}"
            )

        except Exception as e:

            st.error(
                "The prediction could not be generated."
            )

            st.code(
                str(e)
            )

    st.divider()

    # --------------------------------------------------------
    # PREDICTOR GUIDE
    # --------------------------------------------------------

    st.subheader("Predictor Guide")

    predictor_guide = pd.DataFrame({

        "Predictor": [
            "Bathrooms",
            "Bedrooms",
            "Square Feet",
            "Latitude",
            "Longitude",
            "Has Photo",
            "Allows Cats",
            "Allows Dogs",
            "Has Pool",
            "Has Parking",
            "Has Fee",
            "State"
        ],

        "Description": [
            "Number of bathrooms",
            "Number of bedrooms",
            "Apartment floor area",
            "North-south geographical location",
            "East-west geographical location",
            "Whether the listing contains a photo",
            "Whether cats are allowed",
            "Whether dogs are allowed",
            "Whether a swimming pool is available",
            "Whether parking is available",
            "Whether a rental or application fee is required",
            "State where the apartment is located"
        ]
    })

    st.dataframe(
        predictor_guide,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# PAGE 2 — PROJECT OVERVIEW
# ============================================================

elif page == "Project Overview":

    st.title("🏠 Project Overview")

    st.write(
        """
        This project applies machine learning to predict monthly
        apartment rental prices based on apartment characteristics,
        geographical location and available facilities.
        """
    )

    st.divider()

    st.subheader("Project Objective")

    st.write(
        """
        The objective is to develop and evaluate multiple
        regression models, identify the best-performing model,
        and deploy the selected algorithm through an interactive
        Streamlit application.
        """
    )

    st.divider()

    st.subheader("Models Developed")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("Linear Regression")

    with col2:
        st.info("Decision Tree")

    with col3:
        st.info("Random Forest")

    with col4:
        st.info("Gradient Boosting")

    st.divider()

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    st.subheader("Prepared Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Total Variables",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Predictors Used",
            len(features)
        )

    with col4:

        st.metric(
            "Target Variable",
            "Price"
        )

    st.caption(
        """
        Final prepared dataset: 89,730 observations and
        16 variables.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # MODELLING INFORMATION
    # --------------------------------------------------------

    st.subheader("Modelling Dataset")

    modelling_information = pd.DataFrame({

        "Item": [
            "Total Observations",
            "Training Observations",
            "Testing Observations",
            "Training Percentage",
            "Testing Percentage",
            "Number of Predictors"
        ],

        "Value": [
            f"{len(df):,}",
            "71,784",
            "17,946",
            "80%",
            "20%",
            "12"
        ]
    })

    st.dataframe(
        modelling_information,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.subheader("System Workflow")

    st.markdown(
        """
        **Raw Dataset**  
        ↓

        **Data Understanding**  
        ↓

        **Data Preparation**  
        ↓

        **Model Development**  
        ↓

        **Model Evaluation**  
        ↓

        **Best Model Selection**  
        ↓

        **Deployment and Prediction**
        """
    )


# ============================================================
# PAGE 3 — DATA PREPARATION
# ============================================================

elif page == "Data Preparation":

    st.title("🧹 Data Preparation")

    st.write(
        """
        The raw apartment rental dataset was cleaned and
        transformed before model development.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # FINAL DATA QUALITY
    # --------------------------------------------------------

    st.subheader("Final Data Quality")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Final Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Missing Values",
            int(df.isna().sum().sum())
        )

    with col3:

        st.metric(
            "Duplicate Records",
            int(df.duplicated().sum())
        )

    st.success(
        """
        The final prepared dataset contains 89,730 records,
        with no missing values and no duplicate rows.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PREPARATION STEPS
    # --------------------------------------------------------

    st.subheader("Main Data Preparation Steps")

    preparation_steps = pd.DataFrame({

        "Step": [
            "Data Type Conversion",
            "Missing Value Handling",
            "Target Cleaning",
            "Location Cleaning",
            "Outlier Treatment",
            "Duplicate Removal",
            "Feature Engineering",
            "Categorical Encoding"
        ],

        "Description": [
            "Numeric variables were converted into suitable numerical data types.",
            "Missing numeric values were treated before modelling.",
            "Records with missing rental price were removed.",
            "Records with unavailable latitude or longitude were removed.",
            "Extreme observations were treated to reduce the effect of unrealistic values.",
            "Duplicate apartment records were removed.",
            "Pet, pool, parking and fee information were transformed into binary indicators.",
            "State is One-Hot Encoded inside the modelling pipeline."
        ]
    })

    st.dataframe(
        preparation_steps,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    st.subheader("Final Data Types")

    dtype_df = pd.DataFrame({

        "Variable": df.dtypes.index,

        "Data Type": [
            str(dtype)
            for dtype in df.dtypes.values
        ]
    })

    st.dataframe(
        dtype_df,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        """
        State remains a categorical string variable in the
        prepared dataset. It is converted into numerical
        features through One-Hot Encoding before the data
        enters the regression models.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CLEANED PRICE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Rental Price Distribution After Data Cleaning"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        df["price"].dropna(),
        bins=30
    )

    ax.set_xlabel(
        "Monthly Rental Price (USD)"
    )

    ax.set_ylabel(
        "Number of Listings"
    )

    ax.set_title(
        "Distribution of Monthly Rental Prices After Data Cleaning"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------------
    # FEATURE AVAILABILITY
    # --------------------------------------------------------

    st.subheader(
        "Availability of Selected Apartment Features"
    )

    binary_features = [
        "allows_cats",
        "allows_dogs",
        "has_pool",
        "has_parking",
        "has_fee"
    ]

    feature_counts = (
        df[binary_features]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        feature_counts.index,
        feature_counts.values
    )

    ax.set_xlabel(
        "Apartment Feature"
    )

    ax.set_ylabel(
        "Number of Listings"
    )

    ax.set_title(
        "Availability of Selected Apartment Features"
    )

    plt.xticks(
        rotation=30
    )

    for bar in bars:

        value = int(
            bar.get_height()
        )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# PAGE 4 — DATA EXPLORATION
# ============================================================

elif page == "Data Exploration":

    st.title("📊 Data Exploration")

    st.write(
        """
        This section presents the key patterns identified in
        the apartment rental dataset.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------------------------

    st.subheader("Numerical Summary")

    numerical_columns = [
        "bathrooms",
        "bedrooms",
        "price",
        "square_feet",
        "latitude",
        "longitude"
    ]

    numerical_summary = (
        df[numerical_columns]
        .describe()
        .T
    )

    numerical_summary["Median"] = (
        df[numerical_columns]
        .median()
    )

    numerical_summary["Skewness"] = (
        df[numerical_columns]
        .skew()
    )

    st.dataframe(
        numerical_summary.round(2),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # RENTAL PRICE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "1. Distribution of Apartment Rental Prices"
    )

    price_99 = df[
        "price"
    ].quantile(0.99)

    filtered_price = df[
        df["price"] <= price_99
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        filtered_price["price"],
        bins=50
    )

    ax.set_title(
        "Distribution of Apartment Rental Prices "
        "up to the 99th Percentile"
    )

    ax.set_xlabel(
        "Monthly Rental Price (USD)"
    )

    ax.set_ylabel(
        "Frequency"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        The graph is limited to the 99th percentile so that
        the main rental-price distribution can be examined
        without extreme values dominating the visualisation.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # SIZE VS PRICE
    # --------------------------------------------------------

    st.subheader(
        "2. Apartment Size and Rental Price"
    )

    size_limit = df[
        "square_feet"
    ].quantile(0.99)

    price_limit = df[
        "price"
    ].quantile(0.99)

    scatter_data = df[
        (df["square_feet"] <= size_limit)
        &
        (df["price"] <= price_limit)
    ][
        [
            "square_feet",
            "price"
        ]
    ]

    if len(scatter_data) > 5000:

        scatter_data = (
            scatter_data
            .sample(
                5000,
                random_state=42
            )
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        scatter_data["square_feet"],
        scatter_data["price"],
        alpha=0.4
    )

    ax.set_title(
        "Apartment Size and Rental Price "
        "up to the 99th Percentile"
    )

    ax.set_xlabel(
        "Square Feet"
    )

    ax.set_ylabel(
        "Monthly Rental Price (USD)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        The scatter plot shows the relationship between apartment
        size and monthly rental price after limiting extreme
        observations.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # BEDROOMS AND RENT
    # --------------------------------------------------------

    st.subheader(
        "3. Median Rental Price by Number of Bedrooms"
    )

    bedroom_summary = (
        df.groupby("bedrooms")["price"]
        .agg(
            ["count", "mean", "median"]
        )
        .reset_index()
    )

    bedroom_plot = bedroom_summary[
        bedroom_summary["count"] >= 500
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        bedroom_plot[
            "bedrooms"
        ].astype(str),
        bedroom_plot[
            "median"
        ]
    )

    ax.set_title(
        "Median Rental Price by Number of Bedrooms"
    )

    ax.set_xlabel(
        "Number of Bedrooms"
    )

    ax.set_ylabel(
        "Median Monthly Rental Price (USD)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Only bedroom categories with at least 500 listings are
        displayed to avoid drawing conclusions from very small
        groups.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TOP 5 MAJOR STATES
    # --------------------------------------------------------

    st.subheader(
        "4. Median Rental Price Across Five Major States"
    )

    top_5_states = (
        df["state"]
        .value_counts()
        .head(5)
        .index
    )

    top_5_state_prices = (
        df[
            df["state"].isin(
                top_5_states
            )
        ]
        .groupby("state")[
            "price"
        ]
        .median()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    top_5_state_prices.columns = [
        "State",
        "Median Rental Price"
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        top_5_state_prices[
            "State"
        ],
        top_5_state_prices[
            "Median Rental Price"
        ]
    )

    ax.set_title(
        "Median Monthly Rental Price Across Five Major States"
    )

    ax.set_xlabel(
        "State"
    )

    ax.set_ylabel(
        "Median Monthly Rental Price (USD)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        The five states with the largest number of listings are
        compared using median rental price. This demonstrates
        the influence of geographical location on rental prices.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CORRELATION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "5. Correlation Matrix of Numerical Variables"
    )

    correlation_variables = [
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude",
        "price"
    ]

    correlation_matrix = (
        df[
            correlation_variables
        ]
        .corr()
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    image = ax.imshow(
        correlation_matrix,
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(
        range(
            len(
                correlation_variables
            )
        )
    )

    ax.set_yticks(
        range(
            len(
                correlation_variables
            )
        )
    )

    ax.set_xticklabels(
        correlation_variables,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        correlation_variables
    )

    for i in range(
        len(
            correlation_variables
        )
    ):

        for j in range(
            len(
                correlation_variables
            )
        ):

            ax.text(
                j,
                i,
                f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title(
        "Correlation Matrix of Numerical Variables"
    )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# PAGE 5 — MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title("📈 Model Evaluation")

    st.write(
        """
        The four regression models are evaluated before the
        best-performing model is selected for deployment.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # EVALUATION CRITERIA
    # --------------------------------------------------------

    st.subheader("Evaluation Criteria")

    criteria_df = pd.DataFrame({

        "Metric": [
            "MAE",
            "RMSE",
            "R²"
        ],

        "Meaning": [
            "Average absolute difference between actual and predicted rent",
            "Prediction error with greater penalty for large errors",
            "Proportion of rental-price variation explained by the model"
        ],

        "Better Result": [
            "Lower",
            "Lower",
            "Higher"
        ]
    })

    st.dataframe(
        criteria_df,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # COMPLETE PERFORMANCE TABLE
    # --------------------------------------------------------

    st.subheader(
        "Model Performance Comparison"
    )

    st.dataframe(
        performance_df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        """
        The figures above are the latest results obtained from
        the original trained models.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TESTING MAE
    # --------------------------------------------------------

    st.subheader(
        "1. Testing MAE Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        performance_df["Model"],
        performance_df["Testing MAE"]
    )

    ax.set_title(
        "Comparison of Testing MAE"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Testing MAE (USD)"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        """
        **Decision Tree achieved the lowest Testing MAE
        of USD 235.97.**

        This means Decision Tree produced the smallest average
        absolute prediction error among the four models.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TESTING RMSE
    # --------------------------------------------------------

    st.subheader(
        "2. Testing RMSE Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        performance_df["Model"],
        performance_df["Testing RMSE"]
    )

    ax.set_title(
        "Comparison of Testing RMSE"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Testing RMSE (USD)"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        """
        **Random Forest achieved the lowest Testing RMSE
        of USD 412.32.**

        This indicates that Random Forest performed best when
        larger prediction errors were given greater penalties.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TESTING R²
    # --------------------------------------------------------

    st.subheader(
        "3. Testing R² Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        performance_df["Model"],
        performance_df["Testing R²"]
    )

    ax.set_title(
        "Comparison of Testing R²"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Testing R²"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
        """
        **Random Forest achieved the highest Testing R²
        of 0.7516.**

        This means approximately 75.16% of the variation in
        apartment rental prices was explained by the model.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TRAINING VS TESTING MAE
    # --------------------------------------------------------

    st.subheader(
        "4. Training vs Testing MAE"
    )

    x = np.arange(
        len(
            performance_df
        )
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        x - width / 2,
        performance_df[
            "Training MAE"
        ],
        width,
        label="Training MAE"
    )

    ax.bar(
        x + width / 2,
        performance_df[
            "Testing MAE"
        ],
        width,
        label="Testing MAE"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        performance_df[
            "Model"
        ],
        rotation=15
    )

    ax.set_ylabel(
        "MAE (USD)"
    )

    ax.set_title(
        "Training and Testing MAE Comparison"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Comparing training and testing results helps assess
        whether a model generalises well to unseen data and
        whether overfitting may be present.
        """
    )


# ============================================================
# PAGE 6 — BEST MODEL SELECTION
# ============================================================

elif page == "Best Model Selection":

    st.title("🏆 Best Model Selection")

    st.write(
        """
        The best model is selected after comparing the
        testing performance of all four regression models.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------

    st.subheader(
        "Testing Performance Summary"
    )

    selection_df = performance_df[
        [
            "Model",
            "Testing MAE",
            "Testing RMSE",
            "Testing R²"
        ]
    ].copy()

    st.dataframe(
        selection_df,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # WINNING METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Lowest MAE",
            "235.97",
            help="Decision Tree"
        )

        st.caption(
            "Decision Tree"
        )

    with col2:

        st.metric(
            "Lowest RMSE",
            "412.32",
            help="Random Forest"
        )

        st.caption(
            "Random Forest"
        )

    with col3:

        st.metric(
            "Highest R²",
            "0.7516",
            help="Random Forest"
        )

        st.caption(
            "Random Forest"
        )

    st.divider()

    # --------------------------------------------------------
    # FINAL SELECTION
    # --------------------------------------------------------

    st.success(
        "🏆 Selected Best Model: Random Forest Regressor"
    )

    st.subheader(
        "Why Random Forest Was Selected"
    )

    st.write(
        """
        Although the Decision Tree achieved the lowest
        Testing MAE of **USD 235.97**, Random Forest
        demonstrated the strongest overall predictive
        performance.

        Random Forest achieved:

        - **Testing MAE: USD 245.24**
        - **Lowest Testing RMSE: USD 412.32**
        - **Highest Testing R²: 0.7516**

        The lower RMSE indicates better control of larger
        prediction errors, while the higher R² indicates that
        Random Forest explains a greater proportion of the
        variation in apartment rental prices.

        Therefore, Random Forest was selected as the best
        overall model for apartment rental-price prediction.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # RF TRAIN / TEST DETAILS
    # --------------------------------------------------------

    st.subheader(
        "Random Forest Performance"
    )

    rf_summary = pd.DataFrame({

        "Metric": [
            "MAE (USD)",
            "RMSE (USD)",
            "R²"
        ],

        "Training": [
            192.65,
            285.85,
            0.8820
        ],

        "Testing": [
            245.24,
            412.32,
            0.7516
        ]
    })

    st.dataframe(
        rf_summary,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "Best Random Forest Hyperparameters"
    )

    rf_parameters = pd.DataFrame({

        "Parameter": [
            "n_estimators",
            "max_depth",
            "max_features",
            "min_samples_split",
            "min_samples_leaf"
        ],

        "Best Value": [
            100,
            20,
            "sqrt",
            2,
            1
        ]
    })

    st.dataframe(
        rf_parameters,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Best Cross-Validation RMSE: 424.46"
    )

    st.divider()

    # --------------------------------------------------------
    # DEPLOYMENT NOTE
    # --------------------------------------------------------

    st.subheader(
        "Deployment Approach"
    )

    st.info(
        """
        The original Random Forest model above is used for
        official evaluation and best-model selection.

        A lighter Random Forest version is used for the
        Streamlit live prediction because the original saved
        model file was too large for convenient online
        deployment.

        The lighter deployment model does not replace the
        original evaluation results.
        """
    )