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
# 2. LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("apartments_prepared.csv")


df = load_data()


# ============================================================
# 3. LOAD DEPLOYMENT MODEL
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
    "has_photo",
    "allows_cats",
    "allows_dogs",
    "has_pool",
    "has_parking",
    "has_fee",
    "state"
]

target = "price"


# ============================================================
# 5. ORIGINAL MODEL EVALUATION RESULTS
# ============================================================

performance_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Training MAE": [
        368.98,
        251.12,
        192.65,
        254.37
    ],
    "Testing MAE": [
        367.71,
        262.18,
        245.24,
        259.31
    ],
    "Testing RMSE": [
        590.91,
        437.43,
        412.32,
        428.81
    ],
    "Testing R²": [
        0.4898,
        0.6974,
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
        This prediction is generated using the deployed
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
            help="Total number of bathrooms in the apartment."
        )

        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Total number of bedrooms."
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
            help="North-south geographical position of the apartment."
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-74.0060,
            format="%.6f",
            help="East-west geographical position of the apartment."
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
            ["Yes", "No"],
            help="Whether the listing contains apartment photos."
        )

        has_photo = 1 if has_photo_option == "Yes" else 0

        allows_cats_option = st.selectbox(
            "Cats Allowed",
            ["Yes", "No"],
            help="Whether cats are allowed in the apartment."
        )

        allows_cats = 1 if allows_cats_option == "Yes" else 0

        allows_dogs_option = st.selectbox(
            "Dogs Allowed",
            ["Yes", "No"],
            help="Whether dogs are allowed in the apartment."
        )

        allows_dogs = 1 if allows_dogs_option == "Yes" else 0

        has_pool_option = st.selectbox(
            "Swimming Pool Available",
            ["Yes", "No"],
            help="Whether the property provides a swimming pool."
        )

        has_pool = 1 if has_pool_option == "Yes" else 0

        has_parking_option = st.selectbox(
            "Parking Available",
            ["Yes", "No"],
            help="Whether parking is available."
        )

        has_parking = 1 if has_parking_option == "Yes" else 0

        has_fee_option = st.selectbox(
            "Rental / Application Fee",
            ["Yes", "No"],
            help="Whether the listing includes a rental or application fee."
        )

        has_fee = 1 if has_fee_option == "Yes" else 0

    st.divider()

    # --------------------------------------------------------
    # PREDICTION
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
            "has_photo": [has_photo],
            "allows_cats": [allows_cats],
            "allows_dogs": [allows_dogs],
            "has_pool": [has_pool],
            "has_parking": [has_parking],
            "has_fee": [has_fee],
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

            st.info(
                """
                The estimated rental price was generated using
                the deployed Random Forest model.
                """
            )

        except Exception as e:

            st.error(
                "The prediction could not be generated."
            )

            st.code(str(e))

    st.divider()

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
            "North-south geographic location",
            "East-west geographic location",
            "Whether the listing has a photo",
            "Whether cats are allowed",
            "Whether dogs are allowed",
            "Whether a swimming pool is available",
            "Whether parking is available",
            "Whether a rental/application fee is required",
            "State where the apartment is located"
        ]
    })

    st.dataframe(
        predictor_guide,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.subheader("How the Prediction Works")

    st.write(
        """
        **1. User Input**

        The user enters the apartment characteristics.

        **2. Input Preparation**

        The values are arranged using the same predictor structure
        used during model development.

        **3. Preprocessing**

        The State variable is transformed using One-Hot Encoding
        by the saved model pipeline.

        **4. Random Forest Prediction**

        The processed information is passed to the deployed
        Random Forest model.

        **5. Prediction Output**

        The estimated monthly apartment rental price is displayed.
        """
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
        The objective is to develop and evaluate multiple regression
        models, identify the best-performing model, and deploy the
        selected algorithm through an interactive Streamlit application.
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

    st.subheader("Dataset Summary")

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

    st.divider()

    st.subheader("Selected Predictors")

    predictor_list = pd.DataFrame({
        "Predictor": features
    })

    st.dataframe(
        predictor_list,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.subheader("System Workflow")

    st.markdown(
        """
        **Raw Dataset**  
        ↓

        **Data Preparation**  
        ↓

        **Prepared Dataset**  
        ↓

        **Data Exploration**  
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
        Data preparation was performed to improve data quality
        and ensure the dataset was suitable for regression modelling.
        """
    )

    st.divider()

    st.subheader("Main Preparation Steps")

    preparation_steps = pd.DataFrame({
        "Step": [
            "Data Type Conversion",
            "Missing Value Handling",
            "Target Cleaning",
            "Location Cleaning",
            "Outlier Treatment",
            "Duplicate Removal",
            "Binary Feature Creation",
            "Categorical Encoding"
        ],
        "Description": [
            "Numeric variables were converted to appropriate numeric data types.",
            "Missing numeric values were treated using appropriate methods.",
            "Records with missing rental price were removed.",
            "Records with missing latitude or longitude were removed.",
            "Extreme values were treated to reduce the effect of unrealistic observations.",
            "Duplicate apartment records were removed.",
            "Listing characteristics such as photo, pets, pool, parking and fee were represented as binary variables.",
            "State remains categorical in the prepared data and is One-Hot Encoded in the modelling pipeline."
        ]
    })

    st.dataframe(
        preparation_steps,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.subheader("Prepared Dataset Quality")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prepared Records",
            f"{len(df):,}"
        )

    with col2:
        missing_values = int(df.isnull().sum().sum())

        st.metric(
            "Missing Values",
            missing_values
        )

    with col3:
        duplicate_count = int(df.duplicated().sum())

        st.metric(
            "Duplicate Records",
            duplicate_count
        )

    st.divider()

    st.subheader("Data Types Used for Modelling")

    modelling_types = pd.DataFrame({
        "Variable Group": [
            "Numerical",
            "Binary",
            "Categorical",
            "Target"
        ],
        "Variables": [
            "bathrooms, bedrooms, square_feet, latitude, longitude",
            "has_photo, allows_cats, allows_dogs, has_pool, has_parking, has_fee",
            "state",
            "price"
        ],
        "Treatment": [
            "Passed directly to the model",
            "Represented using 0 and 1",
            "One-Hot Encoded before modelling",
            "Numerical regression target"
        ]
    })

    st.dataframe(
        modelling_types,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        """
        The State predictor is stored as a categorical string
        variable in the prepared dataset. It is not passed directly
        into the regression model. One-Hot Encoding converts State
        into numerical features during preprocessing.
        """
    )


# ============================================================
# PAGE 4 — DATA EXPLORATION
# ============================================================

elif page == "Data Exploration":

    st.title("📊 Data Exploration")

    st.write(
        """
        This section explores important characteristics and
        relationships in the prepared apartment rental dataset.
        """
    )

    st.divider()

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Variables",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Average Rent",
            f"${df['price'].mean():,.2f}"
        )

    with col4:
        st.metric(
            "Average Size",
            f"{df['square_feet'].mean():,.0f} sq ft"
        )

    st.divider()

    st.subheader("Prepared Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # RENTAL PRICE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("1. Rental Price Distribution")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        df["price"].dropna(),
        bins=30
    )

    ax.set_title(
        "Distribution of Monthly Apartment Rental Prices"
    )

    ax.set_xlabel(
        "Monthly Rental Price ($)"
    )

    ax.set_ylabel(
        "Number of Listings"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        This graph shows the overall distribution of apartment
        rental prices and identifies the most common rental ranges.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # BEDROOM PRICE
    # --------------------------------------------------------

    st.subheader(
        "2. Average Rental Price by Number of Bedrooms"
    )

    bedroom_price = (
        df.groupby("bedrooms")["price"]
        .mean()
        .reset_index()
        .sort_values("bedrooms")
    )

    bedroom_price = bedroom_price[
        bedroom_price["bedrooms"] <= 10
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        bedroom_price["bedrooms"].astype(str),
        bedroom_price["price"]
    )

    ax.set_title(
        "Average Rental Price by Number of Bedrooms"
    )

    ax.set_xlabel(
        "Number of Bedrooms"
    )

    ax.set_ylabel(
        "Average Monthly Rent ($)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        This chart examines whether apartments with more bedrooms
        generally have higher average rental prices.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # SIZE VS RENT
    # --------------------------------------------------------

    st.subheader(
        "3. Apartment Size vs Rental Price"
    )

    scatter_data = df[
        ["square_feet", "price"]
    ].dropna()

    if len(scatter_data) > 5000:

        scatter_data = scatter_data.sample(
            5000,
            random_state=42
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        scatter_data["square_feet"],
        scatter_data["price"],
        alpha=0.35
    )

    ax.set_title(
        "Apartment Size vs Monthly Rental Price"
    )

    ax.set_xlabel(
        "Apartment Size (Square Feet)"
    )

    ax.set_ylabel(
        "Monthly Rental Price ($)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        The scatter plot shows the relationship between apartment
        floor area and monthly rental price.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # STATE RENT
    # --------------------------------------------------------

    st.subheader(
        "4. Top 10 States by Average Rental Price"
    )

    state_summary = (
        df.groupby("state")["price"]
        .agg(
            Average_Price="mean",
            Listings="count"
        )
        .reset_index()
    )

    state_summary = state_summary[
        state_summary["Listings"] >= 20
    ]

    top_states = (
        state_summary
        .sort_values(
            "Average_Price",
            ascending=False
        )
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        top_states["state"],
        top_states["Average_Price"]
    )

    ax.invert_yaxis()

    ax.set_title(
        "Top 10 States by Average Apartment Rental Price"
    )

    ax.set_xlabel(
        "Average Monthly Rental Price ($)"
    )

    ax.set_ylabel(
        "State"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Rental prices vary across states, indicating that location
        is an important factor in apartment rental-price prediction.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader(
        "5. Correlation Among Numerical Variables"
    )

    correlation_columns = [
        "price",
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude"
    ]

    correlation = df[
        correlation_columns
    ].corr()

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        correlation,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(correlation.columns))
    )

    ax.set_yticks(
        range(len(correlation.columns))
    )

    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        correlation.columns
    )

    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):

            ax.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
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

    st.write(
        """
        Correlation values closer to 1 indicate stronger positive
        relationships, while values closer to -1 indicate stronger
        negative relationships.
        """
    )


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
            "Proportion of variation in rental price explained by the model"
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

    st.subheader("Model Comparison Table")

    st.dataframe(
        performance_df,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    st.subheader("1. Testing MAE Comparison")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        performance_df["Model"],
        performance_df["Testing MAE"]
    )

    ax.set_title(
        "Testing MAE Comparison"
    )

    ax.set_ylabel(
        "Testing MAE"
    )

    plt.xticks(rotation=15)

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Random Forest achieved the lowest Testing MAE of
        **245.24**, indicating the smallest average prediction error.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    st.subheader("2. Testing RMSE Comparison")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        performance_df["Model"],
        performance_df["Testing RMSE"]
    )

    ax.set_title(
        "Testing RMSE Comparison"
    )

    ax.set_ylabel(
        "Testing RMSE"
    )

    plt.xticks(rotation=15)

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Random Forest achieved the lowest Testing RMSE of
        **412.32**, indicating the strongest performance when
        larger prediction errors are penalised.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    st.subheader("3. Testing R² Comparison")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        performance_df["Model"],
        performance_df["Testing R²"]
    )

    ax.set_title(
        "Testing R² Comparison"
    )

    ax.set_ylabel(
        "Testing R²"
    )

    plt.xticks(rotation=15)

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Random Forest achieved the highest Testing R² of
        **0.7516**, meaning that approximately **75.16% of the
        variation in apartment rental prices** is explained by
        the model.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TRAIN VS TEST
    # --------------------------------------------------------

    st.subheader(
        "4. Training vs Testing MAE"
    )

    x = np.arange(
        len(performance_df)
    )

    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        performance_df["Training MAE"],
        width,
        label="Training MAE"
    )

    ax.bar(
        x + width / 2,
        performance_df["Testing MAE"],
        width,
        label="Testing MAE"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        performance_df["Model"],
        rotation=15
    )

    ax.set_title(
        "Training and Testing MAE Comparison"
    )

    ax.set_ylabel(
        "MAE"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Comparing training and testing errors helps identify
        possible overfitting and evaluates how well each model
        generalises to unseen apartment listings.
        """
    )


# ============================================================
# PAGE 6 — BEST MODEL SELECTION
# ============================================================

elif page == "Best Model Selection":

    st.title("🏆 Best Model Selection")

    st.write(
        """
        The best model is selected only after comparing all
        four regression models using the testing results.
        """
    )

    st.divider()

    st.subheader("Model Ranking")

    ranking_df = performance_df[
        [
            "Model",
            "Testing MAE",
            "Testing RMSE",
            "Testing R²"
        ]
    ].copy()

    ranking_df["Overall Result"] = [
        "Lowest Performance",
        "Good",
        "Best",
        "Second Best"
    ]

    st.dataframe(
        ranking_df,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Lowest Testing MAE",
            "245.24"
        )

    with col2:
        st.metric(
            "Lowest Testing RMSE",
            "412.32"
        )

    with col3:
        st.metric(
            "Highest Testing R²",
            "0.7516"
        )

    st.success(
        "🏆 Selected Best Model: Random Forest Regressor"
    )

    st.subheader("Selection Justification")

    st.write(
        """
        Random Forest was selected because it achieved the
        strongest testing performance across all three major
        regression evaluation metrics.

        - **Testing MAE = 245.24** — lowest among all models.
        - **Testing RMSE = 412.32** — lowest among all models.
        - **Testing R² = 0.7516** — highest among all models.

        Therefore, the Random Forest algorithm demonstrated the
        strongest overall ability to predict rental prices for
        unseen apartment listings.
        """
    )

    st.divider()

    st.subheader("Deployment Approach")

    st.info(
        """
        The original Random Forest model was used for official
        model evaluation and best-model selection.

        A lighter Random Forest version is used in the online
        Streamlit application because the original saved model
        file was too large for convenient GitHub and Streamlit
        deployment.

        The lighter model is used only for the live prediction
        function and does not replace the original evaluation results.
        """
    )