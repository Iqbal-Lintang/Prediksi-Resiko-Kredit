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
import time
from fpdf import FPDF
import base64
from datetime import datetime

# TITLE Set page title and configuration
st.set_page_config(
    page_title="Lendora - AI Powered Risk Assessment",
    page_icon="https://i.imgur.com/8RKgXU5.png",
    layout="wide"
)

# LOGO FAVICON Define logo URL
logo_url = "https://i.imgur.com/8RKgXU5.png"

# LOGO AND TITLE Display logo and title in a row
col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo_url, width=100)
with col2:
    # Title
    st.title("Lendora – AI Powered Risk Assessment")

# LOAD MODEL FROM GOOGLE DRIVE Function to load model from Google Drive (same as original)
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

# LOAD MODEL FROM DOWNLOAD DIRECT Alternative approach using direct download (same as original)
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

# Lora AI CONTEXT AWARE LLM CHATBOT Context-aware chatbot function (same as original)
def chatbot_with_context(risk_data=None, key_suffix="default"):
    st.header("Lora AI - Asisten Virtual")
    
    # 6.1 Initialize conversation in session state with a unique key for each instance
    message_key = f"messages_{key_suffix}"
    if message_key not in st.session_state:
        st.session_state[message_key] = [
            {"role": "assistant", "content": "Halo! Saya Lora AI, asisten virtual Anda. Apa yang ingin Anda ketahui tentang calon debitur?"}
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
                    max_tokens=500,
                    temperature=0.9,
                    system="Anda adalah asisten pinjaman bernama Lora AI dari aplikasi bernama Lendora yang memprediksi resiko calon debitur yang memberikan saran tentang penilaian risiko kredit menggunakan dataset dari India dan mata uang Rupee India INR.",
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

# DOWNLOAD REPORT FUNCTION

def create_prediction_pdf(risk_data, input_data):
    """
    Create a PDF report of prediction results and input data
    
    Parameters:
    risk_data (dict): Dictionary containing risk assessment results
    input_data (dict): Dictionary containing all input parameters used for prediction
    
    Returns:
    bytes: PDF file as bytes that can be downloaded
    """
    # Create PDF object
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=30)  # Ensure proper page breaks
    pdf.add_page()
    
    # Add header and title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, 'Laporan Penilaian Resiko Kredit', 0, 1, 'C')
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(190, 10, f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    pdf.cell(0, 10, 'Laporan Ini Dibuat Secara Otomatis Oleh Lendora AI. Disarankan untuk tetap review lebih lanjut oleh Credit Analyst.', 0, 1, 'C')
    pdf.cell(0, 10, 'CONFIDENTIAL - FOR INTERNAL USE ONLY', 0, 1, 'C')
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)
    
    # Add risk prediction result
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Hasil Penilaian Resiko Kredit', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    # Risk score and prediction
    pdf.set_font('Arial', 'B', 12)
    risk_text = "HIGH RISK - Tidak Direkomendasikan Untuk Approval Kredit" if risk_data['risk_prediction'] == 1 else "LOW RISK - Direkomendasikan Untuk Approval Kredit"
    pdf.cell(60, 10, 'Penilaian Resiko:', 0, 0, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(130, 10, risk_text, 0, 1, 'L')
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(60, 10, 'Skor Resiko:', 0, 0, 'L')
    pdf.set_font('Arial', '', 12)
    pdf.cell(130, 10, f"{risk_data['risk_score']:.2%}", 0, 1, 'L')
    pdf.ln(5)
    
    # Risk factors
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 10, 'Faktor Resiko Utama', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    for factor in risk_data['risk_factors']:
        pdf.cell(10, 8, '-', 0, 0, 'L')
        pdf.multi_cell(180, 8, factor, 0, 'L')
    pdf.ln(5)
    
    # Applicant Information Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Informasi Calon Debitur', 0, 1, 'L')
    pdf.ln(2)
    
    # Personal Information
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 8, 'Informasi Pribadi', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    # Two-column layout for personal info
    pdf.cell(60, 8, 'Age:', 0, 0, 'L')
    pdf.cell(40, 8, f"{risk_data['Age']}", 0, 0, 'L')
    pdf.cell(60, 8, 'Marital Status:', 0, 0, 'L')
    pdf.cell(40, 8, f"{risk_data['marital_status']}", 0, 1, 'L')
    
    pdf.cell(60, 8, 'Profession:', 0, 0, 'L')
    pdf.cell(40, 8, f"{risk_data['profession']}", 0, 0, 'L')
    pdf.cell(60, 8, 'Experience:', 0, 0, 'L')
    pdf.cell(40, 8, f"{risk_data['experience']} years", 0, 1, 'L')
    
    pdf.cell(60, 8, 'Income:', 0, 0, 'L')
    # Replace rupee symbol with INR abbreviation
    pdf.cell(130, 8, f"INR {risk_data['Income']:,}", 0, 1, 'L')
    pdf.ln(5)
    
    # Stability Metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(190, 8, 'Stability Metrics', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    pdf.cell(60, 8, 'Job Stability:', 0, 0, 'L')
    pdf.cell(40, 8, f"{risk_data['job_stability']:.4f}", 0, 0, 'L')
    pdf.cell(60, 8, 'Home Stability:', 0, 0, 'L')
    pdf.cell(40, 8, f"{risk_data['home_stability']:.4f}", 0, 1, 'L')
    pdf.ln(5)
    
    # Recommendations
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(190, 10, 'Rekomendasi Bisnis', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    if risk_data['risk_prediction'] == 1:
        recommendations = [
            "Tinjau Ulang Jumlah Pinjaman: Pertimbangkan untuk mengurangi jumlah",
            "Minta Jaminan Tambahan: Untuk mengurangi risiko",
            "Verifikasi Dokumen Ekstra: Lakukan pemeriksaan tambahan.",
            "Riwayat Kredit Lengkap: Periksa riwayat kredit yang lengkap.",
            "Pertimbangan Pendapatan Pasangan Jika Ada: Untuk menilai kemampuan bayar"
        ]
    else:
        recommendations = [
            "Proses Approval Standar: Dapat diproses dengan prosedur standar.",
            "Pertimbangkan Penawaran Khusus: Debitur berkualitas baik.",
            "Minimum Dokumentasi: Cukup dengan dokumen standar.",
            "Fast-Track Processing: Dapat dimasukkan dalam jalur cepat.",
            "Cross-selling Opportunity: Pertimbangkan penawaran produk lain."
        ]

    # FIX: Use a different approach to render recommendations
    y_position = pdf.get_y()
    for i, rec in enumerate(recommendations):
        pdf.set_xy(10, y_position + (i * 8))
        pdf.set_font('Arial', '', 12)  # Reset font after changing position
        pdf.cell(10, 8, '-', 0, 0, 'L')
        pdf.multi_cell(180, 8, rec, 0, 'L')
    
    # Move cursor below the recommendations
    pdf.set_y(y_position + (len(recommendations) * 8) + 5)
    
    # Return PDF as bytes
    pdf_output = pdf.output(dest='S')
    
    # Check the type of pdf_output and handle accordingly
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin1')
    elif isinstance(pdf_output, bytes) or isinstance(pdf_output, bytearray):
        return pdf_output
    else:
        return str(pdf_output).encode('latin1')

def get_download_link(risk_data, input_data):
    """
    Generate a download link for the PDF report
    
    Parameters:
    risk_data (dict): Dictionary containing risk assessment results
    input_data (dict): Dictionary containing all input parameters used for prediction
    
    Returns:
    str: HTML link for downloading the PDF
    """
    # Generate PDF
    pdf_bytes = create_prediction_pdf(risk_data, input_data)
    
    # Encode PDF to base64
    b64 = base64.b64encode(pdf_bytes).decode()
    
    # Generate file name with datetime
    file_name = f"loan_risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Create download link
    href = f'<a href="data:application/pdf;base64,{b64}" download="{file_name}" class="download-button">Download Laporan Hasil Prediksi Resiko</a>'
    
    return href
    
# MAIN APP EXECUTE START Main application execution
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
        
    # APP TABS Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "Informasi Debitur", 
        "Metriks Stabilitas", 
        "Lora AI",
        "Dashboard"
    ])

    # Initialize session state for form data
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}

    with tab1:
        # Create 3 columns for better layout
        col1, col2, col3 = st.columns(3)
        
        # PERSONAL INFORMATION TAB 1
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
            
        # FINANCIAL INFORMATION TAB 1    
        with col2:    
            st.subheader("Informasi Finansial")

            #INCOME 
            income = st.slider(
                "Income (Rupee India)", 
                min_value=1000, 
                max_value=15000000, 
                value=st.session_state.form_data.get('Income', 5000000),
                step=1000  # Adjust step size as needed
            )
            
            # Display income with comma separator
            st.markdown(f"**Selected Income:** ₹{income:,}")
            
            # Determine income segment based on income
            if income < 2500000:
                income_segment = "low"
            elif income < 7500000:
                income_segment = "medium"
            else:
                income_segment = "high"
            
            st.markdown(f"**Income Segment:** {income_segment.capitalize()}")

            # HOUSE OWNERSHIP
            house_ownership_options = ["owned", "rented", "norent_noown"]
            house_ownership = st.selectbox("House_Ownership", house_ownership_options,
                                        index=house_ownership_options.index(st.session_state.form_data.get('House_Ownership', 'owned'))
                                        if st.session_state.form_data.get('House_Ownership') in house_ownership_options else 0)
            
            # CAR OWNERSHIP
            car_ownership_options = ["yes", "no"]
            car_ownership = st.selectbox("Car_Ownership", car_ownership_options,
                                        index=car_ownership_options.index(st.session_state.form_data.get('Car_Ownership', 'yes'))
                                        if st.session_state.form_data.get('Car_Ownership') in car_ownership_options else 0)
            
        # LOCATION INFORMATION TAB 1    
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
            
    
    # STABILITY METRICS TAB 2
    with tab2:
        st.subheader("Analisa Metriks Stabilitas")
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

    
    # Lora AI LLM CHATBOT TAB 3
    with tab3:
        # If no prediction has been made yet, show a simple chat interface
        if "risk_data" not in st.session_state:
            st.info("Gunakan asisten ini untuk menjawab pertanyaan umum tentang pinjaman. Setelah melakukan prediksi, asisten akan dapat menjawab pertanyaan spesifik tentang aplikasi Anda.")
            chatbot_with_context(key_suffix="tab4")
        else:
            # If prediction exists, pass the risk data to the chatbot
            chatbot_with_context(st.session_state.risk_data, key_suffix="tab4_with_data")
            

    # DASHBOARD LOOKER TAB 4
    with tab4:
        st.subheader("Dashboard")
        
        # Replace this with your Looker embed URL
        looker_embed_url = "https://lookerstudio.google.com/embed/reporting/b0bd4eb6-c98c-49b6-ac91-51a4a00895aa/page/cMy9E"
    
        # Display the embedded Looker dashboard
        st.components.v1.iframe(looker_embed_url, height=900, scrolling=True)

    # BIG DIVIDER BETWEEN TABS AND PREDICTION BUTTON
    st.markdown("<hr style='border: 1px solid #bbb;'><br>", unsafe_allow_html=True)

    # PREDICTION BUTTON Button to make prediction
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
        
        # DEBUGGING COLLAPSIBLE INPUT Display the input data in a collapsible section for debugging
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
            
            # MAKE PREDICTION Make prediction
            risk_probability = model.predict_proba(input_data)[0, 1]

            # APPLY RISK PENALTIES Apply risk penalties and adjustments (same as original)
            risk_adjustment = 0
            
            # INCOME BASED RISK ADJUSTMENT & PENALTIES Define income-based risk adjustments
            if income < 2500000:
                risk_adjustment += 0.10
            elif 2500000 <= income < 5000000:
                risk_adjustment += 0.05
            elif 5000000 <= income < 7500000:
                risk_adjustment += 0.02
            elif 7500000 <= income < 10000000:
                risk_adjustment += 0
            elif income >= 10000000:
                risk_adjustment -= 0.05
            
            # JOB AND FINANCIAL STABILITIES RISK ADJUSTMENTS Job and financial stability
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
            
            # AGE RISK ADJUSTMENTS Age-based risk adjustment
            if age < 25:
                risk_adjustment += 0.08
            elif 25 <= age < 55:
                risk_adjustment -= 0.03
            elif age >= 55:
                risk_adjustment += 0.05
            
            # FINAL RISK PROBABILITY Final risk probability
            adjusted_risk_probability = min(risk_probability + risk_adjustment, 1.0)
            
            # RISK THRESHOLD (CHANGEABLE DEPEND ON BANK's RISK APPETITE) Risk classification
            risk_threshold = 0.28
            risk_prediction = 1 if adjusted_risk_probability >= risk_threshold else 0
            
            # KEY RISK FACTORS Identify key risk factors
            risk_factors = []
            
            if income < 2500000:
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

            # STORE INFO FOR Lora AI Store risk information in session state for the chatbot to use
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
            
            # RISK EXPLANATION Risk explanation message
            if risk_prediction == 1:
                 risk_factors_message = "Faktor Resiko Utama:\n- " + "\n- ".join(risk_factors)
            else:
                 risk_factors_message = "Tidak Ada Resiko Signifikan Yang Teridentifikasi."
            
            # Display result
            st.header("Hasil Prediksi")
            
            # DISPLAY RESULT Display the results
            st.metric("Probabilitas Resiko", f"{adjusted_risk_probability:.2%}", delta=None, delta_color="off")
            
            # Create columns for the result display
            result_col1,result_col2 = st.columns(2)
            
            with result_col1:
                if risk_prediction == 1:
                    st.error("Resiko Tinggi - Tidak Direkomendasikan untuk Approval")
                else:
                    st.success("Resiko Rendah - Direkomendasikan untuk Approval")
            
            # RISK GAUGE VISUALIZATION Risk gauge visualization
            with result_col1:
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
            
            # RISK FACTORS DETAILS Risk factors detail section
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
            
            # RECOMMENDATION SECTION Add recommendations section
            st.subheader("Rekomendasi")
            
            if risk_prediction == 1:
                st.warning("""
                ### Rekomendasi untuk Kasus Risiko Tinggi
                
                Berdasarkan analisis, aplikasi ini menunjukkan risiko tinggi. Rekomendasi:
                
                1. **Tinjau Ulang Jumlah Pinjaman**: Pertimbangkan untuk mengurangi jumlah.
                2. **Minta Jaminan Tambahan**: Untuk mengurangi risiko.
                3. **Verifikasi Dokumen Ekstra**: Lakukan pemeriksaan tambahan.
                4. **Riwayat Kredit Lengkap**: Periksa riwayat kredit yang lengkap.
                5. **Pertimbangan Pendapatan Pasangan Jika Ada**: Untuk menilai kemampuan bayar.
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
           # DOWNLOAD REPORT BUTTON
            st.subheader("Download Laporan Resiko")
            
        # Create a dictionary of all input parameters for the PDF
            full_input_data = {
                'Age': age,
                'Marital Status': marital_status,
                'Profession': profession,
                'Experience': experience,
                'Income': income,
                'House Ownership': house_ownership,
                'Car Ownership': car_ownership,
                'State': state,
                'City': city,
                'Current House Years': current_house_yrs,
                'Current Job Years': current_job_yrs,
                'Income Segment': income_segment,
                'Age Group': age_group,
                'Home Stability': home_stability,
                'Job Stability': job_stability,
                'Financial Stability': financial_stability
            }
            
        # Generate and display download button
            download_button = get_download_link(st.session_state.risk_data, full_input_data)
            st.markdown(download_button, unsafe_allow_html=True)
            
        # Add instructions for using the report
            st.info("""
                **Cara Menggunakan Laporan:**
                - Laporan PDF berisi seluruh input dan hasil prediksi resiko kredit
                - Laporan digunakan sebagai bukti screening awal calon debitur
                - Kirim laporan ke Credit Analyst sebagai bukti lebih lanjut
                """)
            
            # ADVANCED METRICS Advanced metrics section
            with st.expander("Metrik Lanjutan", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Stabilitas Finansial")
                
                    # Financial Stability Index
                    financial_stability_pct = min(financial_stability / 7500000, 1.0) * 100
                    st.markdown(f"**Financial Stability Index:** {financial_stability_pct:.1f}%")
                    st.progress(financial_stability_pct / 100)
                    st.markdown("<p style='font-size:12px; font-style:italic;'>Menunjukkan seberapa stabil kondisi finansial debitur dibandingkan standar maksimum.</p>", unsafe_allow_html=True)

                
                    # Income to Age Ratio
                    income_age_ratio = income / age if age > 0 else 0
                    st.markdown(f"**Income to Age Ratio:** {income_age_ratio:,.0f}")
                    st.markdown("<p style='font-size:12px; font-style:italic;'>Mengukur potensi penghasilan debitur relatif terhadap usia. Nilai tinggi berarti daya penghasilan lebih kuat.</p>", unsafe_allow_html=True)
                
                with col2:
                    st.subheader("Stabilitas Pekerjaan & Tempat Tinggal")
                
                    # Calculate Employment Stability Index (ESI)
                    employment_stability_index = (current_job_yrs / experience) * 100 if experience > 0 else 0
                    st.markdown(f"**Employment Stability Index:** {employment_stability_index:.1f}%")
                    st.progress(min(employment_stability_index, 100) / 100)
                    st.markdown("<p style='font-size:12px; font-style:italic;'>Menunjukkan stabilitas pekerjaan berdasarkan lama bekerja di tempat saat ini dibandingkan dengan total pengalaman kerja.</p>", unsafe_allow_html=True)

                
                    # Home Stability
                    home_stability_pct = min(home_stability * 100, 100)
                    st.markdown(f"**Home Stability:** {home_stability:.4f}")
                    st.progress(home_stability_pct / 100)
                    st.caption("Mengukur stabilitas tempat tinggal Debitur, misalnya durasi menetap dan kepemilikan rumah.")
                
                    # Experience Level
                    experience_pct = min(experience / 25, 1.0) * 100
                    st.markdown(f"**Experience Level:** {experience_pct:.1f}%")
                    st.progress(experience_pct / 100)
                    st.caption("Menunjukkan seberapa banyak pengalaman kerja Debitur dibandingkan dengan standar maksimal 25 tahun.")
                

            
            # CHATBOT RECOMMENDATION Chatbot recommendation
            st.info("""
            📱 Gunakan asisten AI kami di tab Lora AI untuk informasi lebih lanjut tentang hasil analisis ini 
            dan rekomendasi khusus untuk kasus Anda.
            """)

            # DASHBOARD RECOMENDATION
            st.info("""
            📊 Gunakan tab Dashboard di Lendora untuk mendapatkan informasi tentang riwayat kondisi pasar kredit.
            """)
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("If this error persists, please check the model compatibility or contact support.")
            

    # FOOTER Add footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
    Copyright © 2025 Iqbal Lintang. All Rights Reserved.<br>
    <span style="font-size: 0.8em;">heylintang@gmail.com</span>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # ABOUT Add "About" section
    with st.expander("About This App"):
        st.write("""
        
        #### Penggunaan:
        1. **Verifikasi & Lengkapi Data**
        Pastikan semua informasi pada tab "Informasi Calon Debitur" telah diisi dengan benar.
        2. **Analisis Risiko Kredit**
        Klik tombol "Prediksi Risiko" untuk mendapatkan hasil analisis berdasarkan model prediktif.
        3. **Gunakan Asisten Lora AI**
        Ajukan pertanyaan kepada Lora AI untuk mendapatkan wawasan tambahan atau klarifikasi.
        4. **Dashboard Interaktif**
        Tinjau tren, analisis, dan data historis melalui dashboard visual.
        5. **Download Laporan**
        Simpan hasil analisis dalam format PDF atau CSV untuk dokumentasi lebih lanjut.
        
        #### Catatan:
        Model ini dilatih menggunakan data historis.
        Hasil prediksi bersifat sebagai referensi, bukan satu-satunya faktor dalam pengambilan keputusan.
        Disarankan untuk menggunakan hasil analisis bersama dengan pertimbangan manual dan kebijakan kredit internal.
        """)
