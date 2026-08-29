# ============================================================
# APARTMENT RENTAL PRICE PREDICTION SYSTEM
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Apartment Rental Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROFESSIONAL UI DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F7F9FC;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    html, body, [class*="css"] {
        font-family: Arial, sans-serif;
    }

    h1 {
        color: #172B4D;
        font-weight: 750;
        letter-spacing: -0.5px;
        margin-bottom: 0.4rem;
    }

    h2 {
        color: #1F3A5F;
        font-weight: 700;
    }

    h3 {
        color: #284B73;
        font-weight: 650;
    }

    p {
        color: #3F4D5E;
        line-height: 1.65;
        font-size: 16px;
    }

    [data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 300px;
        width: 300px;
        background-color: #FFFFFF;
        border-right: 1px solid #E5EAF0;
    }

    [data-testid="stSidebar"] > div:first-child {
        width: 300px;
    }

    [data-testid="stSidebar"] h1 {
        color: #163A63;
        font-size: 24px;
    }

    [data-testid="stSidebar"] p {
        font-size: 14px;
        color: #6B778C;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding-top: 8px;
        padding-bottom: 8px;
        font-size: 15.5px;
        color: #344563;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E4E9F0;
        padding: 18px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(30, 55, 90, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #6B778C;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #163A63;
        font-weight: 700;
    }

    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        border: 1px solid #E5EAF0;
        border-radius: 10px;
        overflow: hidden;
    }

    [data-baseweb="input"] {
        border-radius: 8px;
    }

    [data-baseweb="select"] {
        border-radius: 8px;
    }

    .stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 8px;
        border: none;
        font-weight: 650;
        font-size: 16px;
        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(24, 73, 120, 0.15);
    }

    .stDownloadButton > button {
        width: 100%;
        min-height: 46px;
        border-radius: 8px;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        padding: 16px;
        border: 1px solid #E4E9F0;
        border-radius: 10px;
    }

    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-color: #E7EBF0;
    }

    .hero-card {
        background: linear-gradient(
            135deg,
            #173F6B 0%,
            #245A8D 100%
        );
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 25px;
        color: white;
    }

    .hero-card h1 {
        color: white;
        margin: 0;
        font-size: 34px;
    }

    .hero-card p {
        color: #E8F1FA;
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 16px;
    }

    .section-label {
        color: #6B778C;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .section-title {
        color: #183B61;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .success-box {
        background-color: #EDF8F3;
        border-left: 4px solid #2F8F6B;
        padding: 16px 18px;
        border-radius: 8px;
        color: #225E49;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .info-box {
        background-color: #EDF4FB;
        border-left: 4px solid #3E78B2;
        padding: 16px 18px;
        border-radius: 8px;
        color: #245078;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. UI HELPER FUNCTIONS
# ============================================================

def page_header(title, description):
    st.markdown(
        f"""
        <div class="hero-card">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_header(title, label=None):
    if label:
        st.markdown(
            f"""
            <div style="margin-top: 28px; margin-bottom: 12px;">
                <div class="section-label">{label}</div>
                <div class="section-title">{title}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="margin-top: 28px; margin-bottom: 12px;">
                <div class="section-title">{title}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def read_uploaded_csv(uploaded_file):

    encodings = [
        "utf-8",
        "cp1252",
        "latin-1"
    ]

    last_error = None

    for encoding in encodings:

        try:
            uploaded_file.seek(0)

            data = pd.read_csv(
                uploaded_file,
                encoding=encoding,
                sep=None,
                engine="python"
            )

            return data, encoding

        except Exception as e:
            last_error = e

    raise last_error


# ============================================================
# 4. LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "apartments_prepared.csv"
    )


try:
    df = load_data()

except Exception as e:
    st.error(
        "Unable to load apartments_prepared.csv."
    )
    st.code(str(e))
    st.stop()


# ============================================================
# 5. LOAD DEPLOYMENT MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "random_forest_rent_model.pkl"
    )


try:
    deployment_model = load_model()

except Exception:
    deployment_model = None


# ============================================================
# 6. MODEL VARIABLES
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

binary_columns = [
    "allows_cats",
    "allows_dogs",
    "has_pool",
    "has_parking",
    "has_fee",
    "has_photo"
]

valid_states = set(
    df["state"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)


# ============================================================
# 7. MODEL PERFORMANCE RESULTS
# ============================================================

performance_df = pd.DataFrame(
    {
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
    }
)


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.title(
    "🏠 Apartment Rental"
)

st.sidebar.caption(
    "Machine Learning Prediction System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Model Exploration",
        "Model Performance",
        "Batch Prediction",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Target Variable"
)

st.sidebar.markdown(
    "**Monthly Rental Price**"
)


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "Prediction":

    page_header(
        "Apartment Rental Price Prediction",
        "Estimate the monthly rental price of an apartment using property characteristics, location and available facilities."
    )

    st.markdown(
        """
        <div class="info-box">
            <b>How to use:</b>
            Enter the apartment details, select the available facilities
            and state, then click <b>Predict Rental Price</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    section_header(
        "Apartment Information",
        "Prediction Input"
    )

    col1, col2 = st.columns(2)

    with col1:

        bathrooms = st.number_input(
            "Number of Bathrooms",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            format="%.1f",
            help="Total number of bathrooms."
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
            help="Apartment floor area in square feet."
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

        state = st.selectbox(
            "State",
            sorted(valid_states),
            help="State where the apartment is located."
        )

    with col2:

        has_photo_option = st.selectbox(
            "Listing Has Photo",
            ["Yes", "No"]
        )

        has_photo = (
            1 if has_photo_option == "Yes"
            else 0
        )

        allows_cats_option = st.selectbox(
            "Cats Allowed",
            ["Yes", "No"]
        )

        allows_cats = (
            1 if allows_cats_option == "Yes"
            else 0
        )

        allows_dogs_option = st.selectbox(
            "Dogs Allowed",
            ["Yes", "No"]
        )

        allows_dogs = (
            1 if allows_dogs_option == "Yes"
            else 0
        )

        has_pool_option = st.selectbox(
            "Swimming Pool Available",
            ["Yes", "No"]
        )

        has_pool = (
            1 if has_pool_option == "Yes"
            else 0
        )

        has_parking_option = st.selectbox(
            "Parking Available",
            ["Yes", "No"]
        )

        has_parking = (
            1 if has_parking_option == "Yes"
            else 0
        )

        has_fee_option = st.selectbox(
            "Rental / Application Fee",
            ["Yes", "No"]
        )

        has_fee = (
            1 if has_fee_option == "Yes"
            else 0
        )


    st.divider()


    if deployment_model is None:

        st.error(
            "The prediction model could not be loaded."
        )

    else:

        if st.button(
            "Predict Rental Price",
            type="primary",
            use_container_width=True
        ):

            input_data = pd.DataFrame(
                {
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
                }
            )

            try:

                prediction = (
                    deployment_model.predict(
                        input_data
                    )
                )

                predicted_price = prediction[0]

                st.markdown(
                    """
                    <div class="success-box">
                        Prediction completed successfully.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                result_col1, result_col2, result_col3 = (
                    st.columns([1, 2, 1])
                )

                with result_col2:

                    st.metric(
                        "Estimated Monthly Rental Price",
                        f"${predicted_price:,.2f}"
                    )

            except Exception as e:

                st.error(
                    "Unable to generate the prediction."
                )

                st.code(str(e))


    section_header(
        "Predictor Guide",
        "Input Reference"
    )

    with st.expander(
        "View Predictor Guide",
        expanded=False
    ):

        predictor_guide = pd.DataFrame(
            {
                "Predictor": [
                    "Bathrooms",
                    "Bedrooms",
                    "Square Feet",
                    "Latitude",
                    "Longitude",
                    "Cats Allowed",
                    "Dogs Allowed",
                    "Swimming Pool",
                    "Parking",
                    "Fee",
                    "Listing Photo",
                    "State"
                ],

                "Description": [
                    "Number of bathrooms",
                    "Number of bedrooms",
                    "Apartment floor area",
                    "North-south geographical position",
                    "East-west geographical position",
                    "Whether cats are permitted",
                    "Whether dogs are permitted",
                    "Whether a swimming pool is available",
                    "Whether parking is available",
                    "Whether a rental or application fee applies",
                    "Whether the listing contains a photo",
                    "State where the apartment is located"
                ]
            }
        )

        st.dataframe(
            predictor_guide,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# PAGE 2 — MODEL EXPLORATION
# ============================================================

elif page == "Model Exploration":

    page_header(
        "Model Exploration",
        "Explore the dataset and examine relationships between apartment characteristics and monthly rental prices."
    )


    section_header(
        "Dataset Summary",
        "Data Overview"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

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
            "Predictors",
            len(features)
        )

    with col4:
        st.metric(
            "Target",
            "Price"
        )


    section_header(
        "Dataset Preview",
        "Sample Records"
    )

    with st.expander(
        "View Dataset Preview",
        expanded=False
    ):

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


    section_header(
        "Numerical Summary",
        "Descriptive Statistics"
    )

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

    with st.expander(
        "View Numerical Summary",
        expanded=False
    ):

        st.dataframe(
            numerical_summary.round(2),
            use_container_width=True
        )


    # ========================================================
    # GRAPH 1
    # ========================================================

    section_header(
        "1. Rental Price Distribution",
        "Target Variable Analysis"
    )

    with st.expander(
        "View Analysis",
        expanded=False
    ):

        price_limit = (
            df["price"]
            .quantile(0.99)
        )

        price_data = df[
            df["price"] <= price_limit
        ]

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.hist(
            price_data["price"],
            bins=50
        )

        ax.set_title(
            "Distribution of Rental Prices up to the 99th Percentile"
        )

        ax.set_xlabel(
            "Monthly Rental Price (USD)"
        )

        ax.set_ylabel(
            "Number of Listings"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.write(
            """
            This graph is used to understand the distribution of the
            target variable, monthly rental price. Examining the target
            distribution is important because extreme or highly skewed
            rental values may affect regression model performance.

            The graph is limited to the 99th percentile so that a small
            number of unusually high rental prices do not dominate the
            visualisation. This provides a clearer representation of the
            rental-price range covering most apartment listings.
            """
        )


    # ========================================================
    # GRAPH 2
    # ========================================================

    section_header(
        "2. Apartment Size vs Rental Price",
        "Numerical Relationship"
    )

    with st.expander(
        "View Analysis",
        expanded=False
    ):

        price_limit = (
            df["price"]
            .quantile(0.99)
        )

        size_limit = (
            df["square_feet"]
            .quantile(0.99)
        )

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
                scatter_data.sample(
                    5000,
                    random_state=42
                )
            )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

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
            "Monthly Rental Price (USD)"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.write(
            """
            This scatter plot is used to examine whether apartment size
            is associated with rental price and therefore provides useful
            predictive information.

            Square footage represents the physical size of an apartment
            and may contribute to differences in rental value. Both rental
            price and apartment size are limited to the 99th percentile
            to reduce the visual influence of extreme observations and
            make the main relationship easier to identify.
            """
        )


    # ========================================================
    # GRAPH 3
    # ========================================================

    section_header(
        "3. Median Rental Price by Number of Bedrooms",
        "Property Characteristic"
    )

    with st.expander(
        "View Analysis",
        expanded=False
    ):

        bedroom_summary = (
            df
            .groupby(
                "bedrooms"
            )["price"]
            .agg(
                count="count",
                median="median"
            )
            .reset_index()
        )

        bedroom_summary = (
            bedroom_summary[
                bedroom_summary[
                    "count"
                ] >= 500
            ]
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        bars = ax.bar(
            bedroom_summary[
                "bedrooms"
            ].astype(str),

            bedroom_summary[
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

        for bar, value in zip(
            bars,
            bedroom_summary[
                "median"
            ]
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

                bar.get_height(),

                f"${value:,.0f}",

                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.write(
            """
            This graph is used to investigate how rental prices differ
            across bedroom categories. The number of bedrooms represents
            apartment capacity and is an important property characteristic
            that may influence rental value.

            Median rental price is used instead of the mean because it is
            less affected by unusually expensive listings. Only bedroom
            categories containing at least 500 listings are included so
            that comparisons are based on sufficiently represented groups.
            """
        )


    # ========================================================
    # GRAPH 4
    # ========================================================

    section_header(
        "4. Rental Price Across Five Major States",
        "Location Analysis"
    )

    with st.expander(
        "View Analysis",
        expanded=False
    ):

        major_states = (
            df["state"]
            .value_counts()
            .head(5)
            .index
        )

        state_price = (
            df[
                df["state"].isin(
                    major_states
                )
            ]
            .groupby(
                "state"
            )["price"]
            .median()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        state_price.columns = [
            "State",
            "Median Rental Price"
        ]

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        bars = ax.bar(
            state_price["State"],
            state_price["Median Rental Price"]
        )

        ax.set_title(
            "Median Rental Price Across Five Major States"
        )

        ax.set_xlabel(
            "State"
        )

        ax.set_ylabel(
            "Median Monthly Rental Price (USD)"
        )

        for bar, value in zip(
            bars,
            state_price[
                "Median Rental Price"
            ]
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

                bar.get_height(),

                f"${value:,.0f}",

                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.write(
            """
            This graph is used to examine whether geographical location
            contributes to differences in apartment rental prices.
            Location is an important pricing factor because rental markets
            vary between states.

            The five states with the largest number of listings are used
            so that comparisons are based on well-represented groups.
            Median rental price is used to reduce the influence of unusually
            expensive properties within each state.
            """
        )


    # ========================================================
    # GRAPH 5
    # ========================================================

    section_header(
        "5. Rental Price by Apartment Features",
        "Facility Analysis"
    )

    with st.expander(
        "View Analysis",
        expanded=False
    ):

        binary_feature_labels = {
            "allows_cats": "Cats Allowed",
            "allows_dogs": "Dogs Allowed",
            "has_pool": "Swimming Pool",
            "has_parking": "Parking",
            "has_fee": "Fee Required",
            "has_photo": "Listing Photo"
        }

        feature_price_data = []

        for column, label in (
            binary_feature_labels.items()
        ):

            median_without = (
                df.loc[
                    df[column] == 0,
                    "price"
                ]
                .median()
            )

            median_with = (
                df.loc[
                    df[column] == 1,
                    "price"
                ]
                .median()
            )

            feature_price_data.append(
                {
                    "Feature": label,
                    "Without Feature": median_without,
                    "With Feature": median_with
                }
            )

        feature_price_df = (
            pd.DataFrame(
                feature_price_data
            )
        )

        x = np.arange(
            len(
                feature_price_df
            )
        )

        width = 0.35

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.bar(
            x - width / 2,
            feature_price_df[
                "Without Feature"
            ],
            width,
            label="No"
        )

        ax.bar(
            x + width / 2,
            feature_price_df[
                "With Feature"
            ],
            width,
            label="Yes"
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            feature_price_df[
                "Feature"
            ],
            rotation=20
        )

        ax.set_ylabel(
            "Median Monthly Rental Price (USD)"
        )

        ax.set_title(
            "Median Rental Price With and Without Apartment Features"
        )

        ax.legend()

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.write(
            """
            This graph is used to investigate whether apartment facilities
            and listing characteristics are associated with differences
            in rental prices.

            Binary variables such as pet permission, swimming pool,
            parking, fees and listing photos may provide additional
            predictive information beyond apartment size and location.
            Comparing median rental prices between listings with and without
            each feature helps assess whether these variables may contribute
            useful information to the regression models.
            """
        )


    # ========================================================
    # GRAPH 6
    # ========================================================

    section_header(
        "6. Correlation Matrix",
        "Relationship Analysis"
    )

    with st.expander(
        "View Analysis",
        expanded=False
    ):

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
            figsize=(8, 6)
        )

        image = ax.imshow(
            correlation_matrix,
            vmin=-1,
            vmax=1,
            aspect="auto"
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

                value = (
                    correlation_matrix
                    .iloc[i, j]
                )

                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center"
                )

        ax.set_title(
            "Correlation Matrix of Numerical Variables"
        )

        fig.colorbar(
            image,
            ax=ax,
            label="Correlation Coefficient"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.write(
            """
            This correlation matrix is used to examine the strength and
            direction of relationships among numerical predictors and
            monthly rental price.

            The analysis helps identify variables that may provide useful
            predictive information and also reveals relationships between
            the predictors themselves. Understanding these relationships
            supports feature assessment and helps determine whether a
            simple linear model is sufficient to represent the structure
            of the data.
            """
        )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    page_header(
        "Model Performance",
        "Compare the predictive performance and generalisation ability of the four regression models."
    )


    section_header(
        "Evaluation Metrics",
        "Performance Criteria"
    )

    metric_guide = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE",
                "R²"
            ],

            "Meaning": [
                "Average absolute difference between actual and predicted rent",
                "Prediction error that gives greater weight to large errors",
                "Proportion of variation in rental price explained by the model"
            ],

            "Preferred Result": [
                "Lower",
                "Lower",
                "Higher"
            ]
        }
    )

    with st.expander(
        "View Evaluation Metric Guide",
        expanded=False
    ):

        st.dataframe(
            metric_guide,
            hide_index=True,
            use_container_width=True
        )


    section_header(
        "Model Comparison",
        "Overall Results"
    )

    st.dataframe(
        performance_df,
        hide_index=True,
        use_container_width=True
    )


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    section_header(
        "Testing MAE Comparison",
        "Average Prediction Error"
    )

    with st.expander(
        "View MAE Analysis",
        expanded=False
    ):

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        bars = ax.bar(
            performance_df["Model"],
            performance_df["Testing MAE"]
        )

        ax.set_title(
            "Testing MAE by Regression Model"
        )

        ax.set_ylabel(
            "MAE (USD)"
        )

        plt.xticks(
            rotation=15
        )

        for bar, value in zip(
            bars,
            performance_df[
                "Testing MAE"
            ]
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

                bar.get_height(),

                f"{value:.2f}",

                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.write(
            """
            MAE measures the average absolute prediction error.
            Decision Tree records the lowest Testing MAE of
            **USD 235.97**.
            """
        )


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    section_header(
        "Testing RMSE Comparison",
        "Large Error Sensitivity"
    )

    with st.expander(
        "View RMSE Analysis",
        expanded=False
    ):

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        bars = ax.bar(
            performance_df["Model"],
            performance_df["Testing RMSE"]
        )

        ax.set_title(
            "Testing RMSE by Regression Model"
        )

        ax.set_ylabel(
            "RMSE (USD)"
        )

        plt.xticks(
            rotation=15
        )

        for bar, value in zip(
            bars,
            performance_df[
                "Testing RMSE"
            ]
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

                bar.get_height(),

                f"{value:.2f}",

                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.write(
            """
            RMSE gives greater weight to larger prediction errors.
            Random Forest achieves the lowest Testing RMSE of
            **USD 412.32**.
            """
        )


    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    section_header(
        "Testing R² Comparison",
        "Explained Variation"
    )

    with st.expander(
        "View R² Analysis",
        expanded=False
    ):

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        bars = ax.bar(
            performance_df["Model"],
            performance_df["Testing R²"]
        )

        ax.set_title(
            "Testing R² by Regression Model"
        )

        ax.set_ylabel(
            "R²"
        )

        plt.xticks(
            rotation=15
        )

        for bar, value in zip(
            bars,
            performance_df[
                "Testing R²"
            ]
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

                bar.get_height(),

                f"{value:.4f}",

                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.write(
            """
            R² measures how much variation in rental price is explained
            by the model. Random Forest obtains the highest Testing R²
            of **0.7516**, corresponding to approximately **75.16%**
            of the variation in rental prices.
            """
        )


    # --------------------------------------------------------
    # TRAIN VS TEST
    # --------------------------------------------------------

    section_header(
        "Training vs Testing MAE",
        "Generalisation Check"
    )

    with st.expander(
        "View Generalisation Analysis",
        expanded=False
    ):

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
            performance_df["Model"],
            rotation=15
        )

        ax.set_title(
            "Training and Testing MAE"
        )

        ax.set_ylabel(
            "MAE (USD)"
        )

        ax.legend()

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.write(
            """
            Comparing training and testing errors helps assess model
            generalisation. A substantially lower training error than
            testing error may indicate that a model has learned the
            training data more closely than unseen observations.
            """
        )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    section_header(
        "Best Model Selection",
        "Final Evaluation"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(
            "Lowest Testing MAE",
            "235.97"
        )

        st.write(
            "**Decision Tree**"
        )

    with col2:

        st.metric(
            "Lowest Testing RMSE",
            "412.32"
        )

        st.write(
            "**Random Forest**"
        )

    with col3:

        st.metric(
            "Highest Testing R²",
            "0.7516"
        )

        st.write(
            "**Random Forest**"
        )

    st.markdown(
        """
        <div class="success-box">
            <b>Selected Model:</b>
            Random Forest Regressor
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        Decision Tree achieved the lowest MAE, while Random Forest
        achieved the lowest RMSE and highest R². Considering the
        overall testing performance, Random Forest was selected for
        deployment.
        """
    )


    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    if deployment_model is not None:

        section_header(
            "Actual vs Predicted Rental Price",
            "Prediction Diagnostic"
        )

        with st.expander(
            "View Actual vs Predicted Analysis",
            expanded=False
        ):

            try:

                X_diag = (
                    df[
                        features
                    ]
                    .copy()
                )

                y_diag = (
                    df[
                        target
                    ]
                    .copy()
                )

                (
                    X_train_diag,
                    X_test_diag,
                    y_train_diag,
                    y_test_diag
                ) = train_test_split(
                    X_diag,
                    y_diag,
                    test_size=0.20,
                    random_state=42
                )

                deployment_test_pred = (
                    deployment_model.predict(
                        X_test_diag
                    )
                )

                diagnostic_df = (
                    pd.DataFrame(
                        {
                            "Actual":
                                y_test_diag.values,

                            "Predicted":
                                deployment_test_pred
                        }
                    )
                )

                if len(
                    diagnostic_df
                ) > 4000:

                    diagnostic_plot = (
                        diagnostic_df.sample(
                            4000,
                            random_state=42
                        )
                    )

                else:

                    diagnostic_plot = (
                        diagnostic_df
                    )

                fig, ax = plt.subplots(
                    figsize=(8, 6)
                )

                ax.scatter(
                    diagnostic_plot[
                        "Actual"
                    ],

                    diagnostic_plot[
                        "Predicted"
                    ],

                    alpha=0.35
                )

                minimum_value = min(
                    diagnostic_plot[
                        "Actual"
                    ].min(),

                    diagnostic_plot[
                        "Predicted"
                    ].min()
                )

                maximum_value = max(
                    diagnostic_plot[
                        "Actual"
                    ].max(),

                    diagnostic_plot[
                        "Predicted"
                    ].max()
                )

                ax.plot(
                    [
                        minimum_value,
                        maximum_value
                    ],
                    [
                        minimum_value,
                        maximum_value
                    ],
                    linestyle="--"
                )

                ax.set_title(
                    "Actual vs Predicted Rental Prices"
                )

                ax.set_xlabel(
                    "Actual Rental Price (USD)"
                )

                ax.set_ylabel(
                    "Predicted Rental Price (USD)"
                )

                plt.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

                st.write(
                    """
                    This graph is used to visually assess prediction
                    accuracy. Points located closer to the diagonal
                    reference line represent predictions that are
                    closer to actual rental values.
                    """
                )

            except Exception:

                st.info(
                    """
                    Actual-versus-predicted analysis is unavailable
                    for the current deployment model.
                    """
                )


        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        section_header(
            "Random Forest Feature Importance",
            "Model Interpretation"
        )

        with st.expander(
            "View Feature Importance",
            expanded=False
        ):

            try:

                if hasattr(
                    deployment_model,
                    "named_steps"
                ):

                    pipeline_steps = (
                        deployment_model
                        .named_steps
                    )

                    preprocessor = (
                        pipeline_steps[
                            "preprocessor"
                        ]
                    )

                    if "regressor" in (
                        pipeline_steps
                    ):

                        rf_estimator = (
                            pipeline_steps[
                                "regressor"
                            ]
                        )

                    elif "model" in (
                        pipeline_steps
                    ):

                        rf_estimator = (
                            pipeline_steps[
                                "model"
                            ]
                        )

                    else:
                        rf_estimator = None


                    if (
                        rf_estimator
                        is not None
                        and
                        hasattr(
                            rf_estimator,
                            "feature_importances_"
                        )
                    ):

                        try:

                            processed_names = (
                                preprocessor
                                .get_feature_names_out()
                            )

                        except Exception:

                            processed_names = np.array(
                                [
                                    f"Feature {i + 1}"
                                    for i in range(
                                        len(
                                            rf_estimator
                                            .feature_importances_
                                        )
                                    )
                                ]
                            )

                        importance_df = (
                            pd.DataFrame(
                                {
                                    "Feature":
                                        processed_names,

                                    "Importance":
                                        rf_estimator
                                        .feature_importances_
                                }
                            )
                        )

                        importance_df[
                            "Feature"
                        ] = (
                            importance_df[
                                "Feature"
                            ]
                            .astype(str)
                            .str.replace(
                                "cat__",
                                "",
                                regex=False
                            )
                            .str.replace(
                                "remainder__",
                                "",
                                regex=False
                            )
                        )

                        importance_df = (
                            importance_df
                            .sort_values(
                                "Importance",
                                ascending=False
                            )
                            .head(15)
                            .sort_values(
                                "Importance",
                                ascending=True
                            )
                        )

                        fig, ax = plt.subplots(
                            figsize=(8, 6)
                        )

                        ax.barh(
                            importance_df[
                                "Feature"
                            ],
                            importance_df[
                                "Importance"
                            ]
                        )

                        ax.set_title(
                            "Top Random Forest Feature Importances"
                        )

                        ax.set_xlabel(
                            "Feature Importance"
                        )

                        ax.set_ylabel(
                            "Feature"
                        )

                        plt.tight_layout()

                        st.pyplot(
                            fig,
                            use_container_width=True
                        )

                        plt.close(fig)

                        st.write(
                            """
                            Feature importance is used to identify
                            predictors that contribute more strongly
                            to the Random Forest prediction process.

                            This improves model interpretability by
                            indicating which apartment characteristics
                            and location indicators influence the
                            prediction more strongly.
                            """
                        )

                    else:

                        st.info(
                            "Feature importance is unavailable."
                        )

                else:

                    st.info(
                        "Feature importance is unavailable."
                    )

            except Exception:

                st.info(
                    """
                    Feature importance could not be extracted from
                    the current deployment model.
                    """
                )


    # ========================================================
    # ADVANTAGES AND LIMITATIONS
    # ========================================================

    section_header(
        "Model Advantages and Limitations",
        "Algorithm Review"
    )


    linear_model_info = pd.DataFrame(
        {
            "Advantages": [
                "Simple and easy to understand.",
                "Fast to train and computationally efficient.",
                "Useful as a baseline for evaluating more complex models."
            ],

            "Limitations": [
                "Assumes a linear relationship between predictors and rental price.",
                "May not capture complex interactions between apartment characteristics.",
                "Predictive performance may be weaker when relationships are nonlinear."
            ]
        }
    )


    decision_tree_info = pd.DataFrame(
        {
            "Advantages": [
                "Can capture nonlinear relationships.",
                "Relatively easy to interpret.",
                "Does not require a linear relationship between predictors and target."
            ],

            "Limitations": [
                "Can overfit the training dataset.",
                "Small changes in data can produce different tree structures.",
                "Predictions may be less stable than ensemble methods."
            ]
        }
    )


    random_forest_info = pd.DataFrame(
        {
            "Advantages": [
                "Captures complex and nonlinear relationships.",
                "Combines multiple trees to produce more stable predictions.",
                "Reduces variance compared with a single Decision Tree.",
                "Provides feature-importance information."
            ],

            "Limitations": [
                "Requires more computational resources than simpler models.",
                "Saved model files can become large because many trees are stored.",
                "Less interpretable than Linear Regression or a single Decision Tree.",
                "Training and hyperparameter tuning may take longer."
            ]
        }
    )


    gradient_boosting_info = pd.DataFrame(
        {
            "Advantages": [
                "Can model complex nonlinear relationships.",
                "Sequentially improves predictions by correcting previous errors.",
                "Can achieve strong predictive performance.",
                "Provides flexibility through several tuning parameters."
            ],

            "Limitations": [
                "Can require more training time than simpler models.",
                "Performance is sensitive to hyperparameter settings.",
                "Sequential training is less parallelisable than Random Forest.",
                "More difficult to interpret than simpler regression models."
            ]
        }
    )


    with st.expander(
        "Linear Regression",
        expanded=False
    ):

        st.dataframe(
            linear_model_info,
            hide_index=True,
            use_container_width=True
        )


    with st.expander(
        "Decision Tree Regressor",
        expanded=False
    ):

        st.dataframe(
            decision_tree_info,
            hide_index=True,
            use_container_width=True
        )


    with st.expander(
        "Random Forest Regressor",
        expanded=False
    ):

        st.dataframe(
            random_forest_info,
            hide_index=True,
            use_container_width=True
        )


    with st.expander(
        "Gradient Boosting Regressor",
        expanded=False
    ):

        st.dataframe(
            gradient_boosting_info,
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# PAGE 4 — BATCH PREDICTION
# ============================================================

elif page == "Batch Prediction":

    page_header(
        "Batch Prediction",
        "Upload multiple apartment records and generate rental-price predictions in a single operation."
    )


    section_header(
        "Required Input Format",
        "Upload Requirements"
    )

    with st.expander(
        "View Required Columns",
        expanded=False
    ):

        required_columns_df = (
            pd.DataFrame(
                {
                    "Required Column":
                        features
                }
            )
        )

        st.dataframe(
            required_columns_df,
            hide_index=True,
            use_container_width=True
        )


    st.markdown(
        """
        <div class="info-box">

        <b>CSV Input Format</b><br><br>

        Binary variables must use:<br>

        <b>1 = Yes</b><br>
        <b>0 = No</b><br><br>

        Binary columns:<br>

        allows_cats, allows_dogs, has_pool,
        has_parking, has_fee, has_photo<br><br>

        For <b>state</b>, use the same state abbreviation
        format as the prepared dataset, such as
        CA, TX, FL or NY.<br><br>

        <b>Example row:</b><br>

        1, 2, 900, 34.0522, -118.2437,
        1, 1, 0, 1, 0, 1, CA

        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload Apartment CSV File",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            batch_df, detected_encoding = (
                read_uploaded_csv(
                    uploaded_file
                )
            )


            section_header(
                "Uploaded Data Preview",
                "File Review"
            )

            with st.expander(
                "View Uploaded Data",
                expanded=False
            ):

                st.dataframe(
                    batch_df.head(10),
                    use_container_width=True
                )

                st.write(
                    f"""
                    Detected file encoding:
                    **{detected_encoding}**
                    """
                )


            missing_columns = [
                column
                for column in features
                if column not in batch_df.columns
            ]

            extra_columns = [
                column
                for column in batch_df.columns
                if column not in features
            ]


            if missing_columns:

                st.error(
                    """
                    Prediction cannot continue because the uploaded
                    file is missing required predictor columns.
                    """
                )

                st.write(
                    "**Missing required columns:**"
                )

                st.write(
                    ", ".join(
                        missing_columns
                    )
                )


            else:

                prediction_input = (
                    batch_df[
                        features
                    ]
                    .copy()
                )

                prediction_input[
                    "state"
                ] = (
                    prediction_input[
                        "state"
                    ]
                    .astype("string")
                    .str.strip()
                )


                total_records = len(
                    batch_df
                )


                duplicate_mask = (
                    prediction_input
                    .duplicated(
                        keep=False
                    )
                )

                duplicate_records = int(
                    duplicate_mask.sum()
                )


                missing_value_mask = (
                    prediction_input
                    .isna()
                    .any(
                        axis=1
                    )
                )

                missing_value_records = int(
                    missing_value_mask.sum()
                )


                invalid_binary_mask = (
                    pd.Series(
                        False,
                        index=batch_df.index
                    )
                )

                for column in binary_columns:

                    invalid_binary_mask = (
                        invalid_binary_mask
                        |
                        (
                            prediction_input[
                                column
                            ].notna()
                            &
                            ~prediction_input[
                                column
                            ].isin(
                                [0, 1]
                            )
                        )
                    )

                invalid_binary_records = int(
                    invalid_binary_mask.sum()
                )


                invalid_state_mask = (
                    prediction_input[
                        "state"
                    ].notna()
                    &
                    ~prediction_input[
                        "state"
                    ].astype(str).isin(
                        valid_states
                    )
                )

                invalid_state_records = int(
                    invalid_state_mask.sum()
                )


                invalid_row_mask = (
                    missing_value_mask
                    |
                    invalid_binary_mask
                    |
                    invalid_state_mask
                )

                valid_row_mask = (
                    ~invalid_row_mask
                )

                valid_records = int(
                    valid_row_mask.sum()
                )


                # =================================================
                # DATA QUALITY SUMMARY — KEEP VISIBLE
                # =================================================

                section_header(
                    "Data Quality Summary",
                    "Validation Results"
                )

                quality_col1, quality_col2, quality_col3 = (
                    st.columns(3)
                )

                with quality_col1:

                    st.metric(
                        "Uploaded Records",
                        f"{total_records:,}"
                    )

                    st.metric(
                        "Extra Columns",
                        len(
                            extra_columns
                        )
                    )

                with quality_col2:

                    st.metric(
                        "Duplicate Records",
                        f"{duplicate_records:,}"
                    )

                    st.metric(
                        "Rows with Missing Values",
                        f"{missing_value_records:,}"
                    )

                with quality_col3:

                    st.metric(
                        "Invalid Binary Rows",
                        f"{invalid_binary_records:,}"
                    )

                    st.metric(
                        "Invalid State Rows",
                        f"{invalid_state_records:,}"
                    )

                st.metric(
                    "Valid Rows for Prediction",
                    f"{valid_records:,}"
                )


                with st.expander(
                    "View Data Quality Details",
                    expanded=False
                ):

                    if extra_columns:

                        st.info(
                            """
                            Extra columns were detected.
                            They will remain in the output file
                            but will not be used by the model.
                            """
                        )

                        st.write(
                            "**Extra columns:**"
                        )

                        st.write(
                            ", ".join(
                                extra_columns
                            )
                        )


                    if duplicate_records > 0:

                        st.info(
                            """
                            Duplicate predictor records were detected.
                            They are retained because separate apartment
                            records may legitimately contain identical
                            predictor values.
                            """
                        )


                    if missing_value_records > 0:

                        st.warning(
                            """
                            Rows containing missing required predictor
                            values will not be sent to the prediction model.
                            """
                        )


                    if invalid_binary_records > 0:

                        st.warning(
                            """
                            Some binary variables contain values other
                            than 0 or 1. These rows will not be predicted.
                            """
                        )


                    if invalid_state_records > 0:

                        st.warning(
                            """
                            Some rows contain state values that do not
                            match the values used in the prepared dataset.
                            These rows will not be predicted.
                            """
                        )


                    if (
                        missing_value_records == 0
                        and
                        invalid_binary_records == 0
                        and
                        invalid_state_records == 0
                    ):

                        st.success(
                            """
                            All uploaded rows passed
                            the required data-quality checks.
                            """
                        )


                if deployment_model is None:

                    st.error(
                        """
                        The prediction model could not be loaded.
                        """
                    )


                elif valid_records == 0:

                    st.error(
                        """
                        No valid rows are available for prediction.
                        Please correct the uploaded data.
                        """
                    )


                else:

                    if st.button(
                        "Generate Batch Predictions",
                        type="primary",
                        use_container_width=True
                    ):

                        try:

                            results_df = (
                                batch_df.copy()
                            )

                            results_df[
                                "predicted_monthly_rent"
                            ] = np.nan

                            results_df[
                                "prediction_status"
                            ] = "Predicted"


                            results_df.loc[
                                missing_value_mask,
                                "prediction_status"
                            ] = (
                                "Missing required value"
                            )


                            results_df.loc[
                                invalid_binary_mask,
                                "prediction_status"
                            ] = (
                                "Invalid binary value"
                            )


                            results_df.loc[
                                invalid_state_mask,
                                "prediction_status"
                            ] = (
                                "Invalid state value"
                            )


                            issue_count = (
                                missing_value_mask.astype(int)
                                +
                                invalid_binary_mask.astype(int)
                                +
                                invalid_state_mask.astype(int)
                            )

                            multiple_issue_mask = (
                                issue_count > 1
                            )

                            results_df.loc[
                                multiple_issue_mask,
                                "prediction_status"
                            ] = (
                                "Multiple data quality issues"
                            )


                            valid_prediction_data = (
                                prediction_input.loc[
                                    valid_row_mask,
                                    features
                                ]
                                .copy()
                            )

                            predictions = (
                                deployment_model.predict(
                                    valid_prediction_data
                                )
                            )

                            results_df.loc[
                                valid_row_mask,
                                "predicted_monthly_rent"
                            ] = predictions


                            st.markdown(
                                f"""
                                <div class="success-box">
                                    Predictions generated successfully
                                    for <b>{valid_records:,}</b>
                                    valid apartment record(s).
                                </div>
                                """,
                                unsafe_allow_html=True
                            )


                            section_header(
                                "Prediction Results",
                                "Batch Output"
                            )

                            with st.expander(
                                "View Prediction Results",
                                expanded=True
                            ):

                                st.dataframe(
                                    results_df,
                                    use_container_width=True
                                )


                            section_header(
                                "Prediction Summary",
                                "Output Statistics"
                            )

                            predicted_values = (
                                results_df.loc[
                                    valid_row_mask,
                                    "predicted_monthly_rent"
                                ]
                            )

                            summary_col1, summary_col2, summary_col3 = (
                                st.columns(3)
                            )

                            with summary_col1:

                                st.metric(
                                    "Predicted Records",
                                    f"{valid_records:,}"
                                )

                            with summary_col2:

                                st.metric(
                                    "Average Predicted Rent",
                                    f"${predicted_values.mean():,.2f}"
                                )

                            with summary_col3:

                                st.metric(
                                    "Median Predicted Rent",
                                    f"${predicted_values.median():,.2f}"
                                )


                            csv_output = (
                                results_df
                                .to_csv(
                                    index=False
                                )
                                .encode(
                                    "utf-8"
                                )
                            )

                            st.download_button(
                                label=
                                    "Download Prediction Results",

                                data=
                                    csv_output,

                                file_name=
                                    "apartment_rental_predictions.csv",

                                mime=
                                    "text/csv",

                                use_container_width=True
                            )


                        except Exception as e:

                            st.error(
                                """
                                Unable to generate batch predictions.
                                """
                            )

                            st.code(
                                str(e)
                            )


        except Exception as e:

            st.error(
                """
                Unable to read the uploaded CSV file.
                """
            )

            st.code(
                str(e)
            )


# ============================================================
# PAGE 5 — ABOUT
# ============================================================

elif page == "About":

    page_header(
        "About the Study",
        "Machine learning research for data-driven apartment rental-price estimation."
    )


    section_header(
        "Research Purpose",
        "Study Focus"
    )

    st.write(
        """
        This study investigates the application of machine learning
        techniques to apartment rental-price estimation using
        historical property listing data.

        The research evaluates the predictive capability of several
        regression techniques for apartment rental-price estimation.

        Linear Regression, Decision Tree, Random Forest and Gradient
        Boosting are compared using MAE, RMSE and R² to investigate
        differences in predictive accuracy, error behaviour and
        generalisation performance.

        The comparison also provides insight into whether nonlinear
        and ensemble-learning approaches can capture rental-price
        relationships more effectively than a traditional linear
        baseline model.
        """
    )


    section_header(
        "Business Purpose",
        "Practical Application"
    )

    st.write(
        """
        Apartment rental prices can vary substantially because
        properties differ in size, room configuration, geographical
        location, pet policies, facilities and listing characteristics.

        A machine-learning-based rental-price estimation system can
        support rental platforms, property owners and property managers
        by providing a more consistent and efficient method of estimating
        rental prices from historical listing information.

        For rental platforms, automated rental-price estimation may
        reduce the time required to manually assess listings and support
        more consistent pricing recommendations across a large number
        of properties.
        """
    )


    section_header(
        "Study Dataset",
        "Data Scope"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(
            "Prepared Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Predictors",
            len(features)
        )

    with col3:

        st.metric(
            "Target",
            "Monthly Rental Price"
        )


    section_header(
        "Variables Used in Modelling",
        "Model Inputs"
    )

    with st.expander(
        "View Model Variables",
        expanded=False
    ):

        variable_df = pd.DataFrame(
            {
                "Variable Type": [
                    "Numerical",
                    "Binary",
                    "Categorical",
                    "Target"
                ],

                "Variables": [
                    "bathrooms, bedrooms, square_feet, latitude, longitude",
                    "allows_cats, allows_dogs, has_pool, has_parking, has_fee, has_photo",
                    "state",
                    "price"
                ],

                "Treatment": [
                    "Used as numerical predictors",
                    "Represented using 0 and 1",
                    "One-Hot Encoded during model preprocessing",
                    "Monthly rental price to be predicted"
                ]
            }
        )

        st.dataframe(
            variable_df,
            hide_index=True,
            use_container_width=True
        )


    section_header(
        "Regression Techniques Investigated",
        "Machine Learning Methods"
    )

    with st.expander(
        "View Regression Techniques",
        expanded=False
    ):

        models_df = pd.DataFrame(
            {
                "Model": [
                    "Linear Regression",
                    "Decision Tree Regressor",
                    "Random Forest Regressor",
                    "Gradient Boosting Regressor"
                ],

                "Role in the Study": [
                    "Provides a baseline for evaluating more complex models.",
                    "Examines nonlinear relationships using decision rules.",
                    "Examines ensemble learning through multiple decision trees.",
                    "Examines sequential boosting for improved predictive performance."
                ]
            }
        )

        st.dataframe(
            models_df,
            hide_index=True,
            use_container_width=True
        )