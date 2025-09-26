import streamlit as st
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Diet Recommendation System",
    page_icon="🍎",
    layout="wide"
)

# --- Custom CSS for Styling ---
def apply_custom_styles():
    """Applies custom CSS for buttons and titles."""
    st.markdown(
        """
        <style>
        .stButton > button {
            width: 100%;
            border-radius: 0.5rem;
            border: 1px solid #FF4B4B;
            background-color: #FF4B4B;
            color: white;
        }
        .stButton > button:hover {
            background-color: #FFFFFF;
            color: #FF4B4B;
            border: 1px solid #FF4B4B;
        }
        h1 {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_styles()

# --- Load Data ---
@st.cache_data
def load_data():
    """Loads clustered food data from CSV file."""
    return pd.read_csv("clustered_food_data.csv")

df = load_data()

# --- Helper Functions ---
def calculate_bmi(weight, height):
    """Calculates BMI from weight (kg) and height (cm)."""
    height_m = height / 100
    return weight / (height_m ** 2)

def get_bmi_category(bmi):
    """Returns BMI category and target cluster."""
    if bmi < 18.5:
        return "Underweight", "Weight Gain"
    elif bmi < 25:
        return "Healthy Weight", "Healthy"
    else:
        return "Overweight", "Weight Loss"

def create_meal_plan(food_df):
    """Splits 9 random foods into breakfast, lunch, dinner."""
    selected = food_df.sample(n=9, replace=False)
    return selected.iloc[0:3], selected.iloc[3:6], selected.iloc[6:9]

def display_meal_plan(breakfast, lunch, dinner):
    """Displays meal plan in 3 columns."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🍳 Breakfast")
        for _, row in breakfast.iterrows():
            st.markdown(f"- **{row['food'].title()}** ({row['Caloric Value']:.0f} kcal)")
    with col2:
        st.subheader("☀️ Lunch")
        for _, row in lunch.iterrows():
            st.markdown(f"- **{row['food'].title()}** ({row['Caloric Value']:.0f} kcal)")
    with col3:
        st.subheader("🌙 Dinner")
        for _, row in dinner.iterrows():
            st.markdown(f"- **{row['food'].title()}** ({row['Caloric Value']:.0f} kcal)")

# --- Main Application ---
st.title("🍎 DIET RECOMMENDATION SYSTEM 🥗")

# Input Section
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170, step=1)
with col2:
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=200.0, value=65.0, step=0.1)
    food_pref = st.selectbox("Food Preference", options=["veg", "non-veg"], index=0)

# Recommend Button
if st.button("RECOMMEND", key="recommend_button"):
    st.markdown("<hr>", unsafe_allow_html=True)

    # Calculate BMI and category
    bmi = calculate_bmi(weight, height)
    bmi_category, recommendation_cluster = get_bmi_category(bmi)

    st.info(f"Your BMI is **{bmi:.2f}** ({bmi_category}). "
            f"Based on this, we recommend a diet for **{recommendation_cluster}**.")
    st.markdown("---")

    # Filter foods
    filtered_df = df[(df["cluster_label"] == recommendation_cluster) & (df["food_type"] == food_pref)]

    if len(filtered_df) >= 9:
        breakfast, lunch, dinner = create_meal_plan(filtered_df)
        display_meal_plan(breakfast, lunch, dinner)
    else:
        st.warning(f"Not enough unique '{food_pref}' food options in the '{recommendation_cluster}' category. "
                   f"Please try a different preference.")
