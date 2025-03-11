import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
from PIL import Image
import re
import io
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import requests
import gdown
import anthropic
import os
import sys
import re
import json
from typing import Dict, Any, Optional

# Set page title and configuration
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
    st.title("Aplikasi Prediksi Resiko Debitur dengan OCR")

# Function to load model from Google Drive (same as original)
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

# Alternative approach using direct download (same as original)
@st.cache_resource
def load_model_from_direct_link():
    """Load model from direct download link if available"""
    try:
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
            return None
    except Exception as e:
        st.error(f"Error with direct download: {e}")
        raise e

# Context-aware chatbot function (same as original)
def chatbot_with_context(risk_data=None, key_suffix="default"):
    st.header("CrediBot - Asisten Virtual")

    # Initialize conversation in session state with a unique key for each instance
    message_key = f"messages_{key_suffix}"
    if message_key not in st.session_state:
        st.session_state[message_key] = [
            {"role": "assistant", "content": "Halo! Saya CrediBot, asisten virtual Anda. Apa yang ingin Anda ketahui tentang calon debitur?"}
        ]

    # Display chat history
    for message in st.session_state[message_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input - add a unique key to prevent duplicate widget ID
    prompt = st.chat_input("Ketik pesan Anda di sini...", key=f"chat_input_{key_suffix}")

    if prompt:
        # Add user message to session
        st.session_state[message_key].append({"role": "user", "content": prompt})

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
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "your_api_key_here")

        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)

        # Generate response using Claude 3 Haiku
        with st.spinner("Memproses jawaban..."):
            try:
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=400,
                    temperature=0.8,
                    system="Anda adalah asisten pinjaman dari aplikasi prediksi resiko calon debitur yang memberikan saran tentang penilaian risiko kredit menggunakan dataset dari India dan mata uang INR.",
                    messages=[{"role": "user", "content": f"{context}\n{prompt}"}]
                )
                answer = response.content[0].text

            except Exception as e:
                answer = f"Terjadi kesalahan: {e}"

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Save response to session history
        st.session_state[message_key].append({"role": "assistant", "content": answer})

# Main application execution
if __name__ == "__main__":
    # Try to load the model
    try:
        # Show a message while loading
        with st.spinner("Loading model... This may take a moment."):
            # First try the direct link method if configured in secrets
            direct_model = load_model_from_direct_link()
            
            # If direct link didn't work, use Google Drive method
            if direct_model is None:
                model = load_model_from_gdrive()
            else:
                model = direct_model
        
        st.success("Model Sukses Dimasukan!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("""
        Failed to load the model. Please check that:
        1. The Google Drive link is accessible/shareable
        2. You have the required packages installed (gdown)
        3. Your internet connection is stable
        
        If you're running this locally, try installing gdown: `pip install gdown`
        """)
        st.stop()

    # Create tabs for better organization - now four tabs including OCR
    tab1, tab2, tab3 = st.tabs(["Informasi Calon Debitur", "Stability Metrics", "CrediBot"])

    # Initialize session state for form data
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}

    with tab1:
        # Create 3 columns for better layout
        col1, col2, col3 = st.columns(3)
        
        # Personal Information
        with col1:
            st.subheader("Informasi Pribadi")
            age = st.number_input("Age", min_value=17, max_value=120, 
                                value=st.session_state.form_data.get('Age', 35))
            
            # Create age group based on input age
            if age < 40:
                age_group = "middle age"
            else:
                age_group = "old"
                
            marital_status = st.selectbox("Married/Single", ["married", "single"], 
                                        index=0 if st.session_state.form_data.get('Married/Single') == "married" else 1)
            
            profession_options = [
                "mechanical_engineer", "software_developer", "technical_writer", 
                "civil_servant", "librarian", "economist", "flight_attendant", 
                "architect", "designer", "physician", "financial_analyst", 
                "air_traffic_controller", "police_officer", "artist", "engineer",
                "lawyer", "consultant", "teacher", "doctor", "other"
            ]
            profession = st.selectbox("Profession", profession_options,
                                    index=profession_options.index(st.session_state.form_data.get('Profession', 'other')) 
                                    if st.session_state.form_data.get('Profession') in profession_options else 19)
            
            experience = st.number_input("Experience", min_value=0, max_value=100, 
                                        value=st.session_state.form_data.get('Experience', 5))
            
        # Financial Information    
        with col2:
            st.subheader("Informasi Finansial")
            income = st.number_input("Income (Rupee India)", min_value=0, max_value=15000000, 
                                    value=st.session_state.form_data.get('Income', 5000000))
            
            # Determine income segment based on income
            if income < 2500000:
                income_segment = "low"
            elif income < 7500000:
                income_segment = "medium"
            else:
                income_segment = "high"
                
            house_ownership_options = ["owned", "rented", "norent_noown"]
            house_ownership = st.selectbox("House_Ownership", house_ownership_options,
                                        index=house_ownership_options.index(st.session_state.form_data.get('House_Ownership', 'owned'))
                                        if st.session_state.form_data.get('House_Ownership') in house_ownership_options else 0)
            
            car_ownership_options = ["yes", "no"]
            car_ownership = st.selectbox("Car_Ownership", car_ownership_options,
                                        index=car_ownership_options.index(st.session_state.form_data.get('Car_Ownership', 'yes'))
                                        if st.session_state.form_data.get('Car_Ownership') in car_ownership_options else 0)
            
        # Location Information    
        with col3:
            st.subheader("Informasi Tempat Tinggal")
            
            state_options = [
                "madhya_pradesh", "maharashtra", "kerala", "odisha", "tamil_nadu",
                "gujarat", "rajasthan", "telangana", "bihar", "andhra_pradesh",
                "west_bengal", "haryana", "puducherry", "karnataka",
                "uttar_pradesh", "himachal_pradesh", "punjab", "tripura",
                "uttarakhand", "jharkhand", "delhi", "chandigarh"
            ]
            
            # Find index of state from form data or default to 0
            state_index = 0
            if 'STATE' in st.session_state.form_data and st.session_state.form_data['STATE'] in state_options:
                state_index = state_options.index(st.session_state.form_data['STATE'])
                
            state = st.selectbox("STATE", state_options, index=state_index)
            
            city_options = [
                "mumbai", "delhi_city", "bangalore", "hyderabad", "chennai", 
                "kolkata", "jaipur", "pune", "ahmedabad", "lucknow", "new_delhi", 
                "patna", "bhopal", "indore", "thane", "nagpur", "ghaziabad",
                "agra", "vadodara", "meerut", "rajkot", "amritsar", "varanasi"
            ]
            
            # Find index of city from form data or default to 0
            city_index = 0
            if 'CITY' in st.session_state.form_data and st.session_state.form_data['CITY'] in city_options:
                city_index = city_options.index(st.session_state.form_data['CITY'])
                
            city = st.selectbox("CITY", city_options, index=city_index)
            
            current_house_yrs = st.number_input("Current House Years", min_value=1, max_value=120, 
                                                value=st.session_state.form_data.get('CURRENT_HOUSE_YRS', 10))
                                                
            current_job_yrs = st.number_input("Current Job Years", min_value=0, max_value=100, 
                                            value=st.session_state.form_data.get('CURRENT_JOB_YRS', 5))

    with tab2:
        st.subheader("Calculated Stability Metrics")
        st.info("Nilai-nilai ini dihitung secara otomatis berdasarkan masukan Anda")
        
        # Calculate job stability as a float value
        job_stability = round(current_job_yrs / (age - 18), 8) if age > 18 else 0
        
        # Calculate home stability as a float value
        home_stability = round(current_house_yrs / age, 8) if age > 0 else 0
        
        # For financial stability, using income as proxy
        financial_stability = float(income / 2)
        
        # Display calculated metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Job Stability", f"{job_stability:.4f}")
        col2.metric("Home Stability", f"{home_stability:.4f}")
        col3.metric("Financial Stability", f"{financial_stability:.2f}")

    with tab3:
        # If no prediction has been made yet, show a simple chat interface
        if "risk_data" not in st.session_state:
            st.info("Gunakan asisten ini untuk menjawab pertanyaan umum tentang pinjaman. Setelah melakukan prediksi, asisten akan dapat menjawab pertanyaan spesifik tentang aplikasi Anda.")
            chatbot_with_context(key_suffix="tab4")
        else:
            # If prediction exists, pass the risk data to the chatbot
            chatbot_with_context(st.session_state.risk_data, key_suffix="tab4_with_data")

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
            'Age': [age]
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
                
                # Check if all expected features are present
                input_cols = list(input_data.columns)
                
                # Make sure all required features are present
                missing_cols = set(model.feature_names_in_) - set(input_cols)
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                    st.stop()
                
                # Handle column order mismatch
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
                
                # Check for missing columns
                missing_cols = set(column_order) - set(input_data.columns)
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                    st.stop()
                
                input_data = input_data[column_order]
            
            # Make prediction
            risk_probability = model.predict_proba(input_data)[0, 1]

            # Apply risk penalties and adjustments (same as original)
            risk_adjustment = 0
            
            # Define income-based risk adjustments
            if income < 1000000:
                risk_adjustment += 0.10
            elif 1000000 <= income < 2500000:
                risk_adjustment += 0.05
            elif 2500000 <= income < 5000000:
                risk_adjustment += 0.02
            elif 5000000 <= income < 7500000:
                risk_adjustment += 0
            elif income >= 7500000:
                risk_adjustment -= 0.05
            
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
                risk_adjustment += 0.08
            elif 25 <= age < 55:
                risk_adjustment -= 0.03
            elif age >= 55:
                risk_adjustment += 0.05
            
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
            
            # Risk explanation message
            if risk_prediction == 1:
                risk_factors_message = "Faktor Resiko Utama:\n- " + "\n- ".join(risk_factors)
            else:
                risk_factors_message = "Tidak Ada Resiko Signifikan Yang Teridentifikasi."
            
            # Display the results
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
            
            # Risk gauge visualization (same as original)
            with result_col2:
                # Create two columns (left is wider)
                gauge_col, right_spacer = st.columns([6, 4])
            
                # Define three columns to center the gauge
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
                    
                    # Add classification text below the score
                    risk_text = "HIGH RISK" if risk_prediction == 1 else "LOW RISK"
                    risk_color = "#d9534f" if risk_prediction == 1 else "#5cb85c"
                    ax.text(np.radians(90), -0.35, risk_text, 
                            ha='center', va='center', fontsize=16, 
                            fontweight='bold', color=risk_color)
                    
                    # Remove all tick marks and labels for clean appearance
                    ax.set_rgrids([])
                    ax.set_thetagrids([])
                    ax.set_yticklabels([])
                    
                    # Remove all spines for modern minimal look
                    ax.spines['polar'].set_visible(False)
                    
                    # Adjust figure to make best use of space
                    plt.tight_layout()
                    plt.subplots_adjust(top=1.1)
                    
                    # Display the gauge
                    st.pyplot(fig)
                    
                    # Additional text below the gauge
                    st.markdown(f"<div style='text-align: center; font-size: 16px;'>Score: <b>{adjusted_risk_probability:.2%}</b></div>", 
                                unsafe_allow_html=True)
            
            # Risk factors detail section
            st.subheader("Detail Analisis Risiko")
            
            # Display risk factors and model details in expandable sections
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("Faktor-Faktor Risiko", expanded=True):
                    if risk_factors:
                        for factor in risk_factors:
                            st.markdown(f"• {factor}")
                    else:
                        st.write("Tidak ada faktor risiko signifikan yang teridentifikasi.")
            
            # Add recommendations section
            st.subheader("Rekomendasi")
            
            if risk_prediction == 1:
                st.warning("""
                ### Rekomendasi untuk Kasus Risiko Tinggi
                
                Berdasarkan analisis, aplikasi ini menunjukkan risiko tinggi. Rekomendasi:
                
                1. **Tinjau Ulang Jumlah Pinjaman**: Pertimbangkan untuk mengurangi jumlah.
                2. **Minta Jaminan Tambahan**: Untuk mengurangi risiko.
                3. **Verifikasi Dokumen Ekstra**: Lakukan pemeriksaan tambahan.
                4. **Riwayat Kredit Lengkap**: Periksa riwayat kredit yang lengkap.
                5. **Pertimbangan Pendapatan Pasangan**: Untuk menilai kemampuan bayar.
                """)
            else:
                st.success("""
                ### Rekomendasi untuk Kasus Risiko Rendah
                
                Berdasarkan analisis, aplikasi ini menunjukkan risiko rendah. Rekomendasi:
                
                1. **Proses Approval Standar**: Dapat diproses dengan prosedur standar.
                2. **Pertimbangkan Penawaran Khusus**: Debitur berkualitas baik.
                3. **Minimum Dokumentasi**: Cukup dengan dokumen standar.
                4. **Fast-Track Processing**: Dapat dimasukkan dalam jalur cepat.
                5. **Cross-selling Opportunity**: Pertimbangkan penawaran produk lain.
                """)
            
            # Advanced metrics section
            with st.expander("Metrik Lanjutan", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Stabilitas Finansial")
                    
                    # Financial stability as percentage of maximum expected value
                    financial_stability_pct = min(financial_stability / 10000000, 1.0) * 100
                    
                    # Create a progress bar for financial stability
                    st.markdown(f"**Financial Stability Index:** {financial_stability_pct:.1f}%")
                    st.progress(financial_stability_pct / 100)
                    
                    # Income to age ratio - a rough metric of earning power
                    income_age_ratio = income / age if age > 0 else 0
                    st.markdown(f"**Income to Age Ratio:** {income_age_ratio:,.0f}")
                
                with col2:
                    st.subheader("Stabilitas Pekerjaan & Tempat Tinggal")
                    
                    # Job stability visualization
                    st.markdown(f"**Job Stability:** {job_stability:.4f}")
                    job_stability_pct = min(job_stability * 100, 100)
                    st.progress(job_stability_pct / 100)
                    
                    # Home stability visualization
                    st.markdown(f"**Home Stability:** {home_stability:.4f}")
                    home_stability_pct = min(home_stability * 100, 100)
                    st.progress(home_stability_pct / 100)
                    
                    # Experience visualization
                    experience_pct = min(experience / 20, 1.0) * 100
                    st.markdown(f"**Experience Level:** {experience_pct:.1f}%")
                    st.progress(experience_pct / 100)
            
            # Chatbot recommendation
            st.info("""
            📱 Gunakan asisten AI kami di tab 'CrediBot' untuk informasi lebih lanjut tentang hasil analisis ini 
            dan rekomendasi khusus untuk kasus Anda.
            """)
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("If this error persists, please check the model compatibility or contact support.")
    

    # Add "About" section
    with st.expander("About This Application"):
        st.write("""
        
        #### Penggunaan:
        1. Verifikasi dan lengkapi data pada tab Informasi Calon Debitur
        2. Klik "Prediksi Resiko" untuk mendapatkan hasil analisis
        3. Gunakan asisten CrediBot untuk informasi tambahan
        
        #### Catatan:
        Model ini dilatih menggunakan data historis dari India dengan mata uang INR.
        Prediksi harus digunakan sebagai alat bantu, bukan penentu keputusan final.
        """)
        
        st.markdown("""
        <div style='text-align: center; margin-top: 20px;'>
            <img src="https://i.imgur.com/HQ6nTcZ.png" width="100">
        </div>
        """, unsafe_allow_html=True)

    # Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Copyright © 2025 Iqbal Lintang. All Rights Reserved.
            heylintang@gmail.com
        </div>
        """, 
        unsafe_allow_html=True
    )
