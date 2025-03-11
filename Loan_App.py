import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import requests
import io
import gdown
import anthropic

# Set page title and configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Loan Risk Prediction Model",
    page_icon="https://i.imgur.com/HQ6nTcZ.png",
    layout="wide"
)

# Define logo URL
logo_url = "https://i.imgur.com/HQ6nTcZ.png"

# Display logo and title in a row
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo_url, width=100)
with col2:
    st.title("Aplikasi Prediksi Resiko Debitur")

# Context-aware chatbot function
def chatbot_with_context(risk_data=None):
    st.header("CrediBot - Asisten Virtual")

    # Initialize conversation in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Saya CrediBot, asisten virtual Anda. Apa yang ingin Anda ketahui tentang calon debitur?"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Ketik pesan Anda di sini...")

    if prompt:
        # Add user message to session
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Construct context if risk data is available
        context = ""
        if risk_data:
            risk_factors_str = ", ".join(risk_data.get('risk_factors', [])) or "Tidak Ada Resiko Signifikan"
            context = f"""
            Informasi aplikasi pinjaman:
            - Usia: {risk_data.get('Age', 'Tidak tersedia')}
            - Pendapatan: {risk_data.get('Income', 'Tidak tersedia')}
            - Status Pernikahan: {risk_data.get('marital_status', 'Tidak tersedia')}
            - Profesi: {risk_data.get('profession', 'Tidak tersedia')}
            - Pengalaman: {risk_data.get('experience', 'Tidak tersedia')} tahun
            - Stabilitas Pekerjaan: {risk_data.get('job_stability', 0.0):.4f}
            - Stabilitas Rumah: {risk_data.get('home_stability', 0.0):.4f}
            - Skor risiko: {risk_data.get('risk_score', 0.0):.2%}
            - Prediksi: {"Risiko Tinggi" if risk_data.get('risk_prediction', 0) == 1 else "Risiko Rendah"}
            - Faktor risiko utama: {risk_factors_str}
            """

        # Load API key from Streamlit secrets
        api_key = st.secrets["ANTHROPIC_API_KEY"]

        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)

        # Generate response using Claude 3 Haiku (Cost-efficient)
        with st.spinner("Memproses jawaban..."):
            try:
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=200,  # Limit to save cost
                    temperature=0.5,
                    system="Anda adalah asisten pinjaman yang memberikan saran tentang penilaian risiko kredit.",
                    messages=[{"role": "user", "content": f"{context}\n{prompt}"}]
                )
                answer = response.content[0].text

            except Exception as e:
                answer = f"Terjadi kesalahan: {e}"

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Save response to session history
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Run chatbot with risk data if available
if __name__ == "__main__":
    chatbot_with_context(st.session_state.get("risk_data", {}))


# Function to download model from Google Drive
@st.cache_resource
def load_model_from_gdrive():
    """Load model from Google Drive"""
    try:
        file_id = "1O6mDaL3ptQSR0YHdMgrCjlNjvJzeVXbc"
        output = "calibrated_risk_prediction_model.pkl"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        with st.spinner("Downloading model from Google Drive... This may take a moment."):
            gdown.download(url, output, quiet=False)
        
        return joblib.load(output)
    except Exception as e:
        st.error(f"Error downloading or loading model: {e}")
        raise e

# Alternative approach using direct download
@st.cache_resource
def load_model_from_direct_link():
    """Load model from direct download link if available"""
    try:
        # If you have a direct download link instead of Google Drive
        direct_url = st.secrets.get("MODEL_DIRECT_URL", "")
        if direct_url:
            with st.spinner("Downloading model... This may take a moment."):
                response = requests.get(direct_url)
                if response.status_code == 200:
                    model_data = io.BytesIO(response.content)
                    return joblib.load(model_data)
                else:
                    st.error(f"Failed to download model: HTTP {response.status_code}")
                    raise Exception(f"Failed to download model: HTTP {response.status_code}")
        else:
            # Don't raise an exception or show error if no URL is provided
            return None
    except Exception as e:
        st.error(f"Error with direct download: {e}")
        raise e

# Try to load the model
try:
    # Show a message while loading
    with st.spinner("Loading model... This may take a moment."):
        # First try the direct link method if you have one configured in secrets
        direct_model = load_model_from_direct_link()
        
        # If direct link didn't work, use Google Drive method
        if direct_model is None:
            model = load_model_from_gdrive()
        else:
            model = direct_model
    
    st.success("Model Sukses Dimasukan!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    
    # Provide detailed instructions for troubleshooting
    st.error("""
    Failed to load the model. Please check that:
    1. The Google Drive link is accessible/shareable
    2. You have the required packages installed (gdown)
    3. Your internet connection is stable
    
    If you're running this locally, try installing gdown: `pip install gdown`
    """)
    st.stop()

# Create input form
st.header("Masukan Informasi Calon Debitur")

# Create tabs for better organization - now three tabs including chatbot
tab1, tab2, tab3 = st.tabs(["Informasi Calon Debitur", "Stability Metrics", "Asisten Virtual"])

with tab1:
    # Create 3 columns for better layout
    col1, col2, col3 = st.columns(3)
    
    # Personal Information
    with col1:
        st.subheader("Informasi Pribadi")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        # Create age group based on input age
        if age < 40:  # Based on your data categories
            age_group = "middle age"
        else:
            age_group = "old"
            
        marital_status = st.selectbox("Married/Single", ["married", "single"])
        
        profession = st.selectbox("Profession", [
            "mechanical_engineer", "software_developer", "technical_writer", 
            "civil_servant", "librarian", "economist", "flight_attendant", 
            "architect", "designer", "physician", "financial_analyst", 
            "air_traffic_controller", "police_officer", "artist", "engineer",
            "lawyer", "consultant", "teacher", "doctor", "other"
        ])
        
        experience = st.number_input("Experience", min_value=0, max_value=20, value=5)
        
    # Financial Information    
    with col2:
        st.subheader("Informasi Finansial")
        income = st.number_input("Income (Rupee India)", min_value=0, max_value=15000000, value=5000000)
        
        # Determine income segment based on income
        if income < 2500000:
            income_segment = "low"
        elif income < 7500000:
            income_segment = "medium"
        else:
            income_segment = "high"
            
        house_ownership = st.selectbox("House_Ownership", ["owned", "rented", "norent_noown"])
        car_ownership = st.selectbox("Car_Ownership", ["yes", "no"])
        
    # Location Information    
    with col3:
        st.subheader("Informasi Tempat Tinggal")
        # Use actual states from the dataset
        state = st.selectbox("STATE", [
            "madhya_pradesh", "maharashtra", "kerala", "odisha", "tamil_nadu",
            "gujarat", "rajasthan", "telangana", "bihar", "andhra_pradesh",
            "west_bengal", "haryana", "puducherry", "karnataka",
            "uttar_pradesh", "himachal_pradesh", "punjab", "tripura",
            "uttarakhand", "jharkhand", "delhi", "chandigarh"
        ])
        
        # Use a subset of actual cities from your dataset
        city = st.selectbox("CITY", [
            "mumbai", "delhi_city", "bangalore", "hyderabad", "chennai", 
            "kolkata", "jaipur", "pune", "ahmedabad", "lucknow", "new_delhi", 
            "patna", "bhopal", "indore", "thane", "nagpur", "ghaziabad",
            "agra", "vadodara", "meerut", "rajkot", "amritsar", "varanasi"
        ])
        
        # Replaced sliders with number input fields
        current_house_yrs = st.number_input("Current House Years", min_value=1, max_value=50, value=11)
        current_job_yrs = st.number_input("Current Job Years", min_value=0, max_value=50, value=3)

with tab2:
    st.subheader("Calculated Stability Metrics")
    st.info("Nilai-nilai ini dihitung secara otomatis berdasarkan masukan Anda")
    
    # Calculate job stability as a float value (using the range from your data)
    job_stability = round(current_job_yrs / (age - 18), 8)  # Simple calculation as example
    
    # Calculate home stability as a float value
    home_stability = round(current_house_yrs / age, 8)  # Simple calculation as example
    
    # For financial stability, using income as proxy
    financial_stability = float(income / 2)  # Simple approximation
    
    # Display calculated metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Job Stability", f"{job_stability:.4f}")
    col2.metric("Home Stability", f"{home_stability:.4f}")
    col3.metric("Financial Stability", f"{financial_stability:.2f}")

with tab3:
    # If no prediction has been made yet, show a simple chat interface
    if "risk_data" not in st.session_state:
        st.info("Gunakan asisten ini untuk menjawab pertanyaan umum tentang pinjaman. Setelah melakukan prediksi, asisten akan dapat menjawab pertanyaan spesifik tentang aplikasi Anda.")
        chatbot_with_context()
    else:
        # If prediction exists, pass the risk data to the chatbot
        chatbot_with_context(st.session_state.risk_data)

# Button to make prediction
if st.button("Prediksi Resiko"):
    # Create a DataFrame with all the required columns - using EXACT column names from the model
    input_data = pd.DataFrame({
        'CURRENT_HOUSE_YRS': [current_house_yrs],
        'financial_stability': [financial_stability],
        'Married/Single': [marital_status],
        'Experience': [experience],
        'CURRENT_JOB_YRS': [current_job_yrs],
        'home_stability': [home_stability],
        'job_stability': [job_stability],
        'Profession': [profession],
        'Car_Ownership': [car_ownership],
        'CITY': [city],
        'age_group': [age_group],
        'Income': [income],
        'House_Ownership': [house_ownership],
        'STATE': [state],
        'income_segment': [income_segment],
        'Age': [age]  # Added Age column
    })
    
    # Display the input data in a collapsible section for debugging
    with st.expander("Lihat Input Data"):
        st.dataframe(input_data)
        st.write("Nama Column (untuk debugging):")
        st.write(list(input_data.columns))
    
    try:
        # Check if model has feature_names_in_ attribute and ensure columns match
        if hasattr(model, 'feature_names_in_'):
            with st.expander("Model Feature Information"):
                st.write("Model expected features:", model.feature_names_in_)
            
            # Check if all expected features are in the right order
            input_cols = list(input_data.columns)
            if list(model.feature_names_in_) != input_cols:
                with st.expander("Feature Order Information"):
                    st.write("Feature order mismatch - reordering columns to match model expectations")
                    st.write("Expected order:", list(model.feature_names_in_))
                    st.write("Provided order:", input_cols)
                
                # Reorder columns to match expected order
                input_data = input_data[model.feature_names_in_]
        else:
            # Manually specified order based on training data
            column_order = [
                'CURRENT_HOUSE_YRS', 'financial_stability', 'Married/Single', 
                'Experience', 'CURRENT_JOB_YRS', 'home_stability', 'job_stability', 
                'Profession', 'Car_Ownership', 'CITY', 'age_group', 'Income', 
                'House_Ownership', 'STATE', 'income_segment', 'Age'
            ]
            input_data = input_data[column_order]
        
        # Make prediction
        risk_probability = model.predict_proba(input_data)[0, 1]

        # Step 2: Apply moderate risk penalties
        risk_adjustment = 0
        
       # Define income-based risk adjustments (lower risk for high-income)
        if income < 1000000:
            risk_adjustment += 0.10  # High risk for very low income
        elif 1000000 <= income < 2500000:
            risk_adjustment += 0.05  # Moderate risk
        elif 2500000 <= income < 5000000:
            risk_adjustment += 0.02  # Low risk
        elif 5000000 <= income < 7500000:
            risk_adjustment += 0  # No impact
        elif income >= 7500000:
            risk_adjustment -= 0.05  # Reduce risk for very high earners
        
        # Job and financial stability
        if job_stability < 0.1:
            risk_adjustment += 0.1
        if home_stability < 0.2:
            risk_adjustment += 0.05
        if house_ownership != "owned":
            risk_adjustment += 0.05
        if car_ownership == "no":
            risk_adjustment += 0.02
        if experience < 3:
            risk_adjustment += 0.03
        
        # Age-based risk adjustment
        if age < 25:
            risk_adjustment += 0.08  # Higher risk for young borrowers
        elif 25 <= age < 55:
            risk_adjustment -= 0.03  # Reduced risk for stable working-age borrowers
        elif age >= 55:
            risk_adjustment += 0.05  # Moderate risk for older borrowers nearing retirement
        
        # Final risk probability
        adjusted_risk_probability = min(risk_probability + risk_adjustment, 1.0)
        
        # Risk classification
        risk_threshold = 0.28
        risk_prediction = 1 if adjusted_risk_probability >= risk_threshold else 0
        
        # Identify key risk factors
        risk_factors = []
        
        if income < 1000000:
            risk_factors.append("Penghasilan rendah")
        if job_stability < 0.1:
            risk_factors.append("Pekerjaan tidak stabil")
        if home_stability < 0.2:
            risk_factors.append("Sering berganti tempat tinggal")
        if house_ownership != "owned":
            risk_factors.append("Tidak memiliki rumah")
        if car_ownership == "no":
            risk_factors.append("Tidak memiliki mobil")
        if experience < 3:
            risk_factors.append("Pengalaman kerja terbatas")
        if age < 25:
            risk_factors.append("Calon debitur muda dengan credit history terbatas")
        elif age >= 55:
            risk_factors.append("Usia lanjut mendekati usia pensiun")

        # Store risk information in session state for the chatbot to use
        st.session_state.risk_data = {
            'Age': age,
            'Income': income,
            'risk_score': adjusted_risk_probability,
            'risk_factors': risk_factors if risk_factors else ["Tidak Ada Resiko Signifikan"],
            'risk_prediction': risk_prediction,
            'marital_status': marital_status,
            'experience': experience,
            'profession': profession,
            'home_stability': home_stability,
            'job_stability': job_stability
        }
        
        # Step 6: Generate risk explanation message
        if risk_prediction == 1:
            risk_factors_message = "Faktor Resiko Utama:\n" + "\n".join(risk_factors)
        else:
            risk_factors_message = "Tidak Ada Resiko Signifikan Yang Teridentifikasi."
        
        # Step 7: Display the results
        st.metric("Probabilitas Resiko", f"{adjusted_risk_probability:.2%}", delta=None, delta_color="off")
        st.write(f"Loan Risk Assessment: {'❌ High Risk' if risk_prediction == 1 else '✅ Low Risk'}")
        st.write(risk_factors_message)
        
        # Display result
        st.header("Hasil Prediksi")
        
        # Create columns for the result display
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if risk_prediction == 1:
                st.error("Resiko Tinggi - Tidak Direkomendasikan untuk Approval")
            else:
                st.success("Resiko Rendah - Direkomendasikan untuk Approval")
                
            #st.metric("Risk Probability", f"{risk_probability:.2%}")
        
        # This code should replace the visualization part in the result_col2 section
        with result_col2:
            # Create columns with specific widths to center the gauge
            left_spacer, gauge_col, right_spacer = st.columns([1, 3, 1])
            with gauge_col:
                # Create a modern, sleek speedometer-style gauge
                fig = plt.figure(figsize=(8, 4.5))
                
                # Use polar projection for the speedometer effect
                ax = fig.add_subplot(111, polar=True)
                
                # Set the limits for a half-circle
                ax.set_thetamin(180)
                ax.set_thetamax(0)
                
                # Create custom colormap with smoother transition: green->yellow->orange->red
                # Adjust colors to emphasize yellow/orange/red regions with threshold at 0.28
                colors = [
                    (0.2, 0.7, 0.2),      # Green (for the lowest risk range)
                    (0.7, 0.9, 0.2),      # Yellow-green transition (around 0.15-0.20)
                    (1.0, 0.9, 0.2),      # Yellow (around 0.25-0.30)
                    (1.0, 0.7, 0.0),      # Orange (around 0.35-0.45)
                    (1.0, 0.5, 0.0),      # Deep orange (around 0.50-0.60)
                    (0.9, 0.3, 0.1),      # Orange-red (around 0.70-0.80)
                    (0.9, 0.1, 0.1)       # Red (highest risk)
                ]
                
                # Create the colormap with positions to control the distribution
                # This allocates more space to yellow/orange/red sections
                positions = [0, 0.15, 0.28, 0.40, 0.60, 0.80, 1.0]
                cmap = mpl.colors.LinearSegmentedColormap.from_list('GreenToRed', list(zip(positions, colors)), N=100)
                
                # Create color bands for the gauge
                theta = np.linspace(np.radians(180), np.radians(0), 100)
                radii = np.ones_like(theta) * 0.85
                width = np.ones_like(theta) * 0.17
                bars = ax.bar(theta, radii, width=width, bottom=0.05, alpha=0.85)
                
                # Assign colors to the bars based on position
                for i, bar in enumerate(bars):
                    bar.set_facecolor(cmap(i/99))
                
                # Add risk threshold marker
                threshold_angle = np.radians(180 - 180 * risk_threshold)
                ax.plot([threshold_angle, threshold_angle], [0.05, 0.90], 'k--', linewidth=2, alpha=0.7)
                
                # Add threshold label with clean positioning
                threshold_text_angle = np.radians(180 - 180 * (risk_threshold - 0.03))
                ax.text(threshold_text_angle, 0.25, f'Threshold\n{risk_threshold}', 
                        ha='center', va='center', fontsize=9, fontweight='light',
                        color='black', alpha=0.8, rotation=threshold_text_angle*-57.3 + 90)
                
                # Add the pointer/needle for adjusted risk probability
                needle_angle = np.radians(180 - 180 * adjusted_risk_probability)
                # Create thicker needle base and thinner needle tip
                ax.plot([needle_angle, needle_angle], [0, 0.2], 'k-', linewidth=4, solid_capstyle='round')
                ax.plot([needle_angle, needle_angle], [0.2, 0.80], 'k-', linewidth=2, solid_capstyle='round')
                # Add a nicer pointer base circle
                ax.plot(needle_angle, 0, 'o', markersize=10, markerfacecolor='#333333', markeredgecolor='#999999')
                
                # Add elegant, modern labels with better positioning
                ax.text(np.radians(170), 0.65, 'Low Risk', ha='right', va='center', 
                        fontsize=12, fontweight='bold', color='#333333')
                ax.text(np.radians(120), 0.65, 'High Risk', ha='left', va='center', 
                        fontsize=12, fontweight='bold', color='#333333')
                
                # Title with score - larger and more prominent
                score_text = f'{adjusted_risk_probability:.1%}'
                ax.text(np.radians(90), -0.2, score_text, 
                        ha='center', va='center', fontsize=24, fontweight='bold', color='#333333')
                ax.text(np.radians(90), -0.35, 'Adjusted Risk Score', 
                        ha='center', va='center', fontsize=12, color='#555555')
                
                # Clean up the chart - remove all axes elements for a cleaner look
                ax.set_axis_off()
                
                # Set background color to transparent
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                
                # Add some padding around the plot
                plt.tight_layout(pad=2.0)
                
                # Display the gauge with extra vertical space before and after
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                st.pyplot(fig)
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        
            # Explanation section
            st.subheader("Penjelasan Faktor Resiko")
            
            factors = []
            
            # Income-based risk factors
            if income < 1000000:
                factors.append("Penghasilan Rendah")
            elif 1000000 <= income < 2500000:
                factors.append("Penghasilan Sedang")
            
            # Job and financial stability
            if job_stability < 0.1:
                factors.append("Stabilitas Pekerjaan Rendah")
            if home_stability < 0.2:
                factors.append("Stabilitas Domisili Rendah")
            if house_ownership != "owned":
                factors.append("Tidak Memiliki Rumah")
            if car_ownership == "no":
                factors.append("Tidak Memiliki Mobil")
            if experience < 3:
                factors.append("Pengalaman Kerja Terbatas")
            
            # Age-based risk factors
            if age < 25:
                factors.append("Calon Debitur Muda Dengan Credit History Sedikit")
            elif age >= 55:
                factors.append("Calon Debitur Tua Mendekati Masa Pensiun")
            
            # Display identified risk factors
            if factors:
                st.markdown("#### Faktor Resiko Utama:")
                for factor in factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("Tidak Ada Resiko Signifikan")

        explanation_text = """
        Penilaian risiko didasarkan pada beberapa faktor termasuk stabilitas keuangan, tingkat pendapatan, stabilitas pekerjaan, stabilitas rumah, kepemilikan aset, umur, dan pengalaman kerja. 
        
        Model ini menggunakan ambang batas 0.28 untuk klasifikasi risiko, yang lebih konservatif daripada ambang batas standar 0.5 untuk meminimalkan hasil negatif palsu.
        """
        
        st.info(explanation_text)
        
        # Add a notice about the chatbot
        st.success("Hasil prediksi telah diproses. Anda sekarang dapat menggunakan Asisten Virtual di tab ketiga untuk mendapatkan informasi lebih lanjut tentang hasil penilaian risiko ini.")
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.markdown("##### Debug Information:")
        st.write("Error details:", str(e))
        st.write("Column mismatch error. Check the exact column names, capitalization, and formatting.")
        st.write("Input data columns:")
        st.write(list(input_data.columns))
        st.write("Required columns:")
        st.write(['CURRENT_HOUSE_YRS', 'financial_stability', 'Married/Single', 'Experience', 'CURRENT_JOB_YRS', 'home_stability', 'job_stability', 'Profession', 'Car_Ownership', 'CITY', 'age_group', 'Income', 'House_Ownership', 'STATE', 'income_segment', 'Age'])
        
        # Additional debugging - compare the two sets
        st.write("Missing columns (if any):")
        required = {'CURRENT_HOUSE_YRS', 'financial_stability', 'Married/Single', 'Experience', 'CURRENT_JOB_YRS', 'home_stability', 'job_stability', 'Profession', 'Car_Ownership', 'CITY', 'age_group', 'Income', 'House_Ownership', 'STATE', 'income_segment', 'Age'}
        provided = set(input_data.columns)
        st.write(list(required - provided))
        
        # Check for whitespace or case issues
        st.write("Check for whitespace or case sensitivity issues:")
        for req_col in required:
            for prov_col in provided:
                if req_col.lower() == prov_col.lower() and req_col != prov_col:
                    st.write(f"Case mismatch: '{req_col}' (required) vs '{prov_col}' (provided)")
                if req_col.strip() == prov_col.strip() and req_col != prov_col:
                    st.write(f"Whitespace mismatch: '{req_col}' (required) vs '{prov_col}' (provided)")
