# ============================================================
# APARTMENT RENTAL PRICE PREDICTION SYSTEM
# Streamlit Deployment Prototype
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


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
# 3. LOAD DEPLOYED RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("random_forest_rent_model.pkl")


model = load_model()


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
# 5. SIDEBAR
# ============================================================

st.sidebar.title("🏠 Apartment Rental")

st.sidebar.write(
    "Apartment Rental Price Prediction System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Data Exploration",
        "Model Performance",
        "Random Forest Details",
        "Rental Price Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning Deployment Prototype"
)

st.sidebar.caption(
    "Selected Algorithm: Random Forest Regressor"
)


# ============================================================
# PAGE 1 — PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":

    st.title("🏠 Apartment Rental Price Prediction")

    st.write(
        """
        This system applies machine learning to estimate the
        monthly rental price of apartments based on apartment
        characteristics, location and available facilities.
        """
    )

    st.divider()

    st.subheader("Project Objective")

    st.write(
        """
        The main objective of this project is to develop a
        machine learning model that can predict apartment rental
        prices based on relevant apartment characteristics.

        Four regression models were developed and evaluated:

        - Linear Regression
        - Decision Tree Regressor
        - Random Forest Regressor
        - Gradient Boosting Regressor

        The best-performing model was selected for deployment.
        """
    )

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

    st.subheader("Selected Model")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Best Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "Testing MAE",
            "245.24"
        )

    with col3:
        st.metric(
            "Testing RMSE",
            "412.32"
        )

    with col4:
        st.metric(
            "Testing R²",
            "0.7516"
        )

    st.success(
        """
        Random Forest was selected because it achieved the
        lowest Testing MAE and RMSE and the highest Testing R²
        among the four regression models.
        """
    )

    st.divider()

    st.subheader("System Workflow")

    st.markdown(
        """
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

        **Random Forest Deployment**  
        ↓

        **Rental Price Prediction**
        """
    )


# ============================================================
# PAGE 2 — DATA EXPLORATION
# ============================================================

elif page == "Data Exploration":

    st.title("📊 Data Exploration")

    st.write(
        """
        This section explores the main characteristics and
        patterns of the prepared apartment rental dataset.
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

    st.caption(
        "The table displays the first 10 records from the prepared dataset."
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
        This chart shows how apartment rental prices are
        distributed across different price ranges.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PRICE BY BEDROOMS
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
        This chart compares average rental prices for apartments
        with different numbers of bedrooms.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # SQUARE FEET VS PRICE
    # --------------------------------------------------------

    st.subheader(
        "3. Apartment Size vs Rental Price"
    )

    scatter_data = df[
        [
            "square_feet",
            "price"
        ]
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
        This scatter plot examines the relationship between
        apartment size and monthly rental price.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PRICE BY STATE
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
        Rental prices vary across states, showing that
        location is an important factor in rental-price prediction.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # CORRELATION MATRIX
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
        Correlation values range from -1 to 1.
        Values closer to 1 indicate a stronger positive relationship,
        while values closer to -1 indicate a stronger negative relationship.
        """
    )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write(
        """
        Four regression models were evaluated using MAE,
        RMSE and R² to identify the most suitable model for
        apartment rental-price prediction.
        """
    )

    st.divider()

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

    st.subheader("Best Model")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Selected Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "Testing MAE",
            "245.24"
        )

    with col3:
        st.metric(
            "Testing RMSE",
            "412.32"
        )

    with col4:
        st.metric(
            "Testing R²",
            "0.7516"
        )

    st.success(
        """
        Random Forest achieved the strongest overall testing
        performance and was therefore selected as the best model.
        """
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
    # TESTING MAE
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

    ax.set_xlabel(
        "Regression Model"
    )

    ax.set_ylabel(
        "Testing MAE"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        MAE measures the average absolute prediction error.
        Lower values indicate better performance.

        **Random Forest achieved the lowest Testing MAE of 245.24.**
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TESTING RMSE
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

    ax.set_xlabel(
        "Regression Model"
    )

    ax.set_ylabel(
        "Testing RMSE"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        RMSE gives greater penalties to larger prediction errors.
        Lower RMSE indicates better predictive performance.

        **Random Forest achieved the lowest Testing RMSE of 412.32.**
        """
    )

    st.divider()

    # --------------------------------------------------------
    # TESTING R²
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

    ax.set_xlabel(
        "Regression Model"
    )

    ax.set_ylabel(
        "Testing R²"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        R² measures the proportion of variation in apartment
        rental prices explained by the model.

        **Random Forest achieved the highest Testing R² of 0.7516.**
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

    ax.set_xticks(
        x
    )

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
        Comparing training and testing MAE helps identify
        possible overfitting. Random Forest shows some difference
        between training and testing performance, but it still
        achieved the strongest overall testing performance.
        """
    )

    st.divider()

    st.subheader(
        "Model Selection Conclusion"
    )

    st.info(
        """
        Random Forest was selected because it achieved:

        • Lowest Testing MAE: 245.24

        • Lowest Testing RMSE: 412.32

        • Highest Testing R²: 0.7516

        Testing performance was prioritised because a deployed
        model must predict new and unseen apartment listings.
        """
    )


# ============================================================
# PAGE 4 — RANDOM FOREST DETAILS
# ============================================================

elif page == "Random Forest Details":

    st.title("🌲 Random Forest Model Details")

    st.write(
        """
        This section provides a closer examination of the
        Random Forest model used in the deployment prototype.
        """
    )

    st.divider()

    X = df[
        features
    ]

    y = df[
        target
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # --------------------------------------------------------
    # DEPLOYED MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Deployed Model Performance Check"
    )

    with st.spinner(
        "Evaluating deployed Random Forest model..."
    ):

        deployed_pred = model.predict(
            X_test
        )

    deployed_mae = mean_absolute_error(
        y_test,
        deployed_pred
    )

    deployed_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            deployed_pred
        )
    )

    deployed_r2 = r2_score(
        y_test,
        deployed_pred
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Deployed MAE",
            f"{deployed_mae:.2f}"
        )

    with col2:
        st.metric(
            "Deployed RMSE",
            f"{deployed_rmse:.2f}"
        )

    with col3:
        st.metric(
            "Deployed R²",
            f"{deployed_r2:.4f}"
        )

    st.caption(
        """
        These values are calculated directly from the Random
        Forest model currently loaded by the Streamlit application.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    st.subheader(
        "1. Actual vs Predicted Rental Prices"
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        y_test,
        deployed_pred,
        alpha=0.35
    )

    minimum = min(
        y_test.min(),
        deployed_pred.min()
    )

    maximum = max(
        y_test.max(),
        deployed_pred.max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum]
    )

    ax.set_title(
        "Actual vs Predicted Rental Prices"
    )

    ax.set_xlabel(
        "Actual Rental Price ($)"
    )

    ax.set_ylabel(
        "Predicted Rental Price ($)"
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        """
        Points closer to the diagonal line indicate more accurate
        predictions, while points further away indicate larger
        prediction errors.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "2. Random Forest Feature Importance"
    )

    try:

        preprocessor = model.named_steps[
            "preprocessor"
        ]

        regressor = model.named_steps[
            "regressor"
        ]

        transformed_features = (
            preprocessor
            .get_feature_names_out()
        )

        cleaned_feature_names = []

        for feature_name in transformed_features:

            feature_name = (
                feature_name
                .replace(
                    "cat__",
                    ""
                )
                .replace(
                    "remainder__",
                    ""
                )
            )

            cleaned_feature_names.append(
                feature_name
            )

        importance_df = pd.DataFrame({
            "Feature": cleaned_feature_names,
            "Importance": regressor.feature_importances_
        })

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
        )

        top_importance = (
            importance_df
            .head(15)
            .sort_values(
                "Importance",
                ascending=True
            )
        )

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        ax.barh(
            top_importance["Feature"],
            top_importance["Importance"]
        )

        ax.set_title(
            "Top 15 Random Forest Feature Importances"
        )

        ax.set_xlabel(
            "Feature Importance"
        )

        ax.set_ylabel(
            "Feature"
        )

        plt.tight_layout()

        st.pyplot(fig)

        st.write(
            """
            Features with higher importance values have a greater
            influence on the Random Forest prediction.
            """
        )

        st.subheader(
            "Feature Importance Table"
        )

        st.dataframe(
            importance_df.head(15),
            hide_index=True,
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            "Feature importance could not be displayed."
        )

        st.code(
            str(e)
        )


# ============================================================
# PAGE 5 — RENTAL PRICE PREDICTION
# ============================================================

elif page == "Rental Price Prediction":

    st.title(
        "🔮 Apartment Rental Price Prediction"
    )

    st.write(
        """
        Enter the apartment characteristics below and the
        deployed Random Forest model will estimate the monthly
        rental price.
        """
    )

    st.divider()

    st.subheader(
        "Apartment Information"
    )

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
            step=0.5
        )

        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )

        square_feet = st.number_input(
            "Apartment Size (Square Feet)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=50
        )

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=40.7128,
            format="%.6f"
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-74.0060,
            format="%.6f"
        )

        state_list = sorted(
            df["state"]
            .dropna()
            .astype(str)
            .unique()
        )

        state = st.selectbox(
            "State",
            state_list
        )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        has_photo_option = st.selectbox(
            "Listing Has Photo",
            [
                "Yes",
                "No"
            ]
        )

        has_photo = (
            1
            if has_photo_option == "Yes"
            else 0
        )

        allows_cats_option = st.selectbox(
            "Cats Allowed",
            [
                "Yes",
                "No"
            ]
        )

        allows_cats = (
            1
            if allows_cats_option == "Yes"
            else 0
        )

        allows_dogs_option = st.selectbox(
            "Dogs Allowed",
            [
                "Yes",
                "No"
            ]
        )

        allows_dogs = (
            1
            if allows_dogs_option == "Yes"
            else 0
        )

        has_pool_option = st.selectbox(
            "Swimming Pool Available",
            [
                "Yes",
                "No"
            ]
        )

        has_pool = (
            1
            if has_pool_option == "Yes"
            else 0
        )

        has_parking_option = st.selectbox(
            "Parking Available",
            [
                "Yes",
                "No"
            ]
        )

        has_parking = (
            1
            if has_parking_option == "Yes"
            else 0
        )

        has_fee_option = st.selectbox(
            "Rental / Application Fee",
            [
                "Yes",
                "No"
            ]
        )

        has_fee = (
            1
            if has_fee_option == "Yes"
            else 0
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICT BUTTON
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

            prediction = model.predict(
                input_data
            )

            predicted_price = (
                prediction[0]
            )

            st.success(
                "Prediction completed successfully."
            )

            st.metric(
                "Estimated Monthly Rental Price",
                f"${predicted_price:,.2f}"
            )

            st.info(
                """
                The estimated rental price was generated by
                the deployed Random Forest Regressor using
                the apartment characteristics entered above.
                """
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
    # HOW PREDICTION WORKS
    # --------------------------------------------------------

    st.subheader(
        "How the Prediction Works"
    )

    st.write(
        """
        **1. User Input**

        The user enters the apartment characteristics.

        **2. Input Preparation**

        The values are organised using the same predictor
        structure used during model development.

        **3. Preprocessing**

        The State variable is automatically transformed using
        One-Hot Encoding by the saved model pipeline.

        **4. Random Forest Prediction**

        The processed values are passed to the trained Random
        Forest Regressor.

        **5. Prediction Output**

        The estimated monthly apartment rental price is
        displayed to the user.
        """
    )