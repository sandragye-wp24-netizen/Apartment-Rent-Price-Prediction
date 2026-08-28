import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Apartment Rental Price Prediction System",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("random_forest_rent_model.pkl")


model = load_model()


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("apartments_prepared.csv")


df = load_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🏠 Apartment Rental System")

st.sidebar.write(
    "Use the navigation menu below to explore the system."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Rental Price Prediction",
        "Data Exploration",
        "Model Performance"
    ]
)


# =========================================================
# PAGE 1: RENTAL PRICE PREDICTION
# =========================================================

if page == "Rental Price Prediction":

    st.title("🏠 Apartment Rental Price Prediction")

    st.write(
        "Enter the apartment information below to estimate "
        "its monthly rental price."
    )

    st.divider()

    st.subheader("Apartment Information")


    # -----------------------------------------------------
    # CREATE TWO COLUMNS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # LEFT COLUMN
    # -----------------------------------------------------

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

        # Use states directly from the dataset
        state_list = sorted(
            df["state"].dropna().astype(str).unique()
        )

        state = st.selectbox(
            "State",
            state_list
        )


    # -----------------------------------------------------
    # RIGHT COLUMN
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # PREDICTION BUTTON
    # -----------------------------------------------------

    if st.button(
        "Predict Rental Price",
        type="primary",
        use_container_width=True
    ):

        # IMPORTANT:
        # Column names must match model training features.

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

            prediction = model.predict(input_data)

            predicted_price = prediction[0]

            st.success(
                "Prediction completed successfully."
            )

            st.metric(
                label="Estimated Monthly Rental Price",
                value=f"${predicted_price:,.2f}"
            )

            st.info(
                "The prediction was generated using the "
                "Random Forest Regressor selected as the "
                "best-performing model."
            )


        except Exception as e:

            st.error(
                "The prediction could not be generated."
            )

            st.write("Error details:")

            st.code(str(e))


    # -----------------------------------------------------
    # PREDICTION EXPLANATION
    # -----------------------------------------------------

    st.divider()

    st.subheader("How the Prediction Works")

    st.write(
        """
        1. The user enters the apartment characteristics.

        2. The information is converted into the same format
        used during model training.

        3. The saved Random Forest pipeline automatically
        preprocesses the input data.

        4. The `state` variable is transformed using
        One-Hot Encoding.

        5. The Random Forest Regressor processes the input
        variables and generates an estimated rental price.

        6. The estimated monthly rental price is displayed
        to the user.
        """
    )


# =========================================================
# PAGE 2: DATA EXPLORATION
# =========================================================

elif page == "Data Exploration":

    st.title("📊 Data Exploration")

    st.write(
        "This section provides an overview of the apartment "
        "rental dataset and explores important patterns "
        "related to rental prices."
    )

    st.divider()


    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    st.subheader("Dataset Overview")

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

        if "price" in df.columns:

            st.metric(
                "Average Rental Price",
                f"${df['price'].mean():,.2f}"
            )


    with col4:

        if "square_feet" in df.columns:

            st.metric(
                "Average Size",
                f"{df['square_feet'].mean():,.0f} sq ft"
            )


    st.divider()


    # =====================================================
    # DATA PREVIEW
    # =====================================================

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    st.divider()


    # =====================================================
    # RENTAL PRICE DISTRIBUTION
    # =====================================================

    st.subheader("Rental Price Distribution")

    if "price" in df.columns:

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.hist(
            df["price"].dropna(),
            bins=30
        )

        ax.set_xlabel("Monthly Rental Price ($)")
        ax.set_ylabel("Number of Listings")
        ax.set_title(
            "Distribution of Apartment Rental Prices"
        )

        st.pyplot(fig)

        st.write(
            """
            This chart shows how apartment rental prices are
            distributed across the dataset. It helps identify
            the most common rental-price ranges and whether
            unusually high or low rental prices are present.
            """
        )


    st.divider()


    # =====================================================
    # PRICE BY BEDROOM
    # =====================================================

    st.subheader("Average Rental Price by Number of Bedrooms")

    if (
        "bedrooms" in df.columns
        and "price" in df.columns
    ):

        bedroom_price = (
            df.groupby("bedrooms")["price"]
            .mean()
            .reset_index()
            .sort_values("bedrooms")
        )

        bedroom_price = bedroom_price[
            bedroom_price["bedrooms"] <= 10
        ]

        st.bar_chart(
            bedroom_price,
            x="bedrooms",
            y="price"
        )

        st.write(
            """
            This chart compares average rental prices across
            apartments with different numbers of bedrooms.
            It can be used to examine whether apartments with
            more bedrooms generally have higher rental prices.
            """
        )


    st.divider()


    # =====================================================
    # SQUARE FEET VS PRICE
    # =====================================================

    st.subheader("Apartment Size vs Rental Price")

    if (
        "square_feet" in df.columns
        and "price" in df.columns
    ):

        scatter_data = df[
            ["square_feet", "price"]
        ].dropna()

        # Sample to keep the Streamlit chart responsive
        if len(scatter_data) > 5000:

            scatter_data = scatter_data.sample(
                5000,
                random_state=42
            )

        st.scatter_chart(
            scatter_data,
            x="square_feet",
            y="price"
        )

        st.write(
            """
            The scatter plot examines the relationship between
            apartment size and monthly rental price. In general,
            larger apartments may have higher rental prices,
            although location and other apartment characteristics
            can also influence rental prices.
            """
        )


    st.divider()


    # =====================================================
    # STATE EXPLORATION
    # =====================================================

    st.subheader("Average Rental Price by State")

    if (
        "state" in df.columns
        and "price" in df.columns
    ):

        state_price = (
            df.groupby("state")["price"]
            .agg(["mean", "count"])
            .reset_index()
        )

        state_price.columns = [
            "State",
            "Average Price",
            "Number of Listings"
        ]

        # Avoid states with extremely few records
        state_price = state_price[
            state_price["Number of Listings"] >= 20
        ]

        state_price = (
            state_price
            .sort_values(
                "Average Price",
                ascending=False
            )
            .head(10)
        )

        st.bar_chart(
            state_price,
            x="State",
            y="Average Price"
        )

        st.write(
            """
            The chart displays the states with the highest
            average apartment rental prices among states with
            sufficient listing records. It demonstrates the
            importance of geographical location in rental-price
            prediction.
            """
        )


    st.divider()


    # =====================================================
    # CORRELATION
    # =====================================================

    st.subheader("Numerical Feature Correlation")

    numeric_columns = [
        "price",
        "bathrooms",
        "bedrooms",
        "square_feet",
        "latitude",
        "longitude"
    ]

    available_numeric = [
        col for col in numeric_columns
        if col in df.columns
    ]


    if len(available_numeric) >= 2:

        correlation = (
            df[available_numeric]
            .corr()
            .round(3)
        )

        st.dataframe(
            correlation,
            use_container_width=True
        )

        st.write(
            """
            The correlation table measures the strength and
            direction of relationships among numerical
            variables. Values closer to 1 or -1 indicate
            stronger relationships, while values closer to
            0 indicate weaker linear relationships.
            """
        )


# =========================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "This section compares the regression models developed "
        "for apartment rental price prediction."
    )

    st.divider()


    # =====================================================
    # PERFORMANCE DATA
    # =====================================================

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


    # =====================================================
    # BEST MODEL SUMMARY
    # =====================================================

    st.subheader("Selected Model")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Best Testing MAE",
            "245.24"
        )


    with col2:

        st.metric(
            "Best Testing RMSE",
            "412.32"
        )


    with col3:

        st.metric(
            "Best Testing R²",
            "0.7516"
        )


    st.success(
        "🏆 Best Model: Random Forest Regressor"
    )

    st.write(
        """
        The Random Forest Regressor was selected for deployment
        because it achieved the **lowest Testing MAE (245.24)**,
        **lowest Testing RMSE (412.32)** and
        **highest Testing R² (0.7516)** among the four
        evaluated regression models.
        """
    )


    st.divider()


    # =====================================================
    # MODEL COMPARISON TABLE
    # =====================================================

    st.subheader("Model Comparison")

    st.dataframe(
        performance_df,
        hide_index=True,
        use_container_width=True
    )


    st.divider()


    # =====================================================
    # TESTING MAE
    # =====================================================

    st.subheader("Testing MAE Comparison")

    mae_chart = performance_df[
        ["Model", "Testing MAE"]
    ].set_index("Model")

    st.bar_chart(mae_chart)

    st.write(
        """
        MAE measures the average absolute difference between
        the actual and predicted rental prices. A lower MAE
        indicates better prediction performance. Random Forest
        achieved the lowest Testing MAE of **245.24**.
        """
    )


    st.divider()


    # =====================================================
    # TESTING RMSE
    # =====================================================

    st.subheader("Testing RMSE Comparison")

    rmse_chart = performance_df[
        ["Model", "Testing RMSE"]
    ].set_index("Model")

    st.bar_chart(rmse_chart)

    st.write(
        """
        RMSE gives greater penalties to larger prediction errors.
        A lower RMSE therefore indicates better model performance.
        Random Forest achieved the lowest Testing RMSE of
        **412.32**.
        """
    )


    st.divider()


    # =====================================================
    # TESTING R2
    # =====================================================

    st.subheader("Testing R² Comparison")

    r2_chart = performance_df[
        ["Model", "Testing R²"]
    ].set_index("Model")

    st.bar_chart(r2_chart)

    st.write(
        """
        R² measures the proportion of variation in rental prices
        explained by the model. A higher value indicates stronger
        predictive performance. Random Forest achieved the highest
        Testing R² of **0.7516**, meaning that it explains
        approximately **75.16% of the variation in apartment
        rental prices** in the testing dataset.
        """
    )


    st.divider()


    # =====================================================
    # TRAINING VS TESTING MAE
    # =====================================================

    st.subheader("Training vs Testing MAE")

    mae_comparison = performance_df[
        [
            "Model",
            "Training MAE",
            "Testing MAE"
        ]
    ].set_index("Model")

    st.bar_chart(mae_comparison)

    st.write(
        """
        Comparing training and testing MAE helps identify
        possible overfitting. Random Forest has a lower
        training MAE than testing MAE, indicating some
        difference between training and unseen-data performance.
        However, it still achieved the strongest overall
        testing performance among the evaluated models.
        """
    )


    st.divider()


    # =====================================================
    # FINAL MODEL INTERPRETATION
    # =====================================================

    st.subheader("Model Selection Interpretation")

    st.write(
        """
        Four regression models were evaluated using MAE,
        RMSE and R². Testing performance was prioritised
        because the deployed system is expected to make
        predictions for new and unseen apartment listings.

        Random Forest produced the lowest Testing MAE and
        RMSE while achieving the highest Testing R².
        Therefore, Random Forest demonstrated the strongest
        overall predictive and generalisation performance
        and was selected as the final model for deployment
        in the Streamlit prototype.
        """
    )


# =========================================================
# SIDEBAR INFORMATION
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Apartment Rental Price Prediction Prototype"
)

st.sidebar.caption(
    "Deployment Model: Random Forest Regressor"
)