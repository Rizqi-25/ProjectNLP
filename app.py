# you need to install all these in your terminal
# pip install streamlit
# pip install scikit-learn
# pip install python-docx
# pip install PyPDF2


import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
svc_model = pickle.load(open('./models/svc.pkl', 'rb'))
tfidf = pickle.load(open('./models/tfidf.pkl', 'rb'))
le = pickle.load(open('./models/encoder.pkl', 'rb'))
knn_model = pickle.load(open('./models/knn.pkl', 'rb'))
rf_model = pickle.load(open('./models/randomforest.pkl', 'rb'))


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)  # Menghapus URL yang ada pada resume lalu menggantinya dengan spasi
    cleanText = re.sub(r'RT|cc', ' ', cleanText)  # Menghapus Kata RT dan CC pada resume lalu menggantinya dengan spasi
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)  # Menghapus hastags lalu menggantinya dengan spasi
    cleanText = re.sub(r'@\S+', ' ', cleanText)  # Menghapus kata yang mengandung "@" seperti email lalu menggantinya dengan spasi
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]', ' ', cleanText)  # Menghapus spesial karakter atau simbol
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)  # Menghapus karakter non-ASCII seperti é, ü, ç, emoji, ©, €, ™, 
    #atau simbol lainnya dan menggantinya dengan spasi
    cleanText = re.sub(r'\s+', ' ', cleanText)  # Menghilankan extra spasi
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def predict_category(input_resume, model_name):
    """
    Predict the category of a resume using the selected model.
    Memprediksi kategori dari resume menggunakan model yang dipilih.
    Parameter:
    - input_resume: str, Text resume yang akan digunakan
    - model_name: str, Model yang akan digunakan untuk prediksi. ("SVC", "KNN", "Random Forest")

    Returns:
    - str, Hasil dari prediksi.
    """
    # Preprocess the input text
    cleaned_text = cleanResume(input_resume)

    # Vectorize the cleaned text
    vectorized_text = tfidf.transform([cleaned_text]).toarray()

    # Select and predict with the chosen model
    if model_name == "SVC":
        predicted_category = svc_model.predict(vectorized_text)
    elif model_name == "KNN":
        predicted_category = knn_model.predict(vectorized_text)
    elif model_name == "Random Forest":
        predicted_category = rf_model.predict(vectorized_text)
    else:
        raise ValueError("Invalid model selection. Choose 'SVC', 'KNN', or 'Random Forest'.")

    # Get the name of the predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]

texts = {
    "en": {
        "title": "Resume Category Prediction App",
        "description": "Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.",
        "upload_label": "Upload a Resume",
        "model_label": "Select a prediction model:",
        "show_text": "Show extracted text",
        "predicted_category": "Predicted Category",
        "selected_model": "Selected Model",
        "success_message": "Successfully extracted the text from the uploaded resume.",
        "error_message": "Error processing the file: ",
        "language_label": "Choose Language"
    },
    "id": {
        "title": "Aplikasi Prediksi Kategori Pekerjaan dari Resume",
        "description": "Unggah resume dalam format PDF, TXT, atau DOCX untuk mendapatkan kategori pekerjaan yang diprediksi.",
        "upload_label": "Unggah Resume",
        "model_label": "Pilih model prediksi:",
        "show_text": "Tampilkan teks yang diambil",
        "predicted_category": "Kategori yang Diprediksi",
        "selected_model": "Model yang Dipilih",
        "success_message": "Berhasil mengambil teks dari resume yang diunggah.",
        "error_message": "Terjadi kesalahan saat memproses file: ",
        "language_label": "Pilih Bahasa"
    }
}

# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    # Pilihan bahasa
    language = st.sidebar.selectbox("Choose Language / Pilih Bahasa", ["English", "Indonesia"])
    lang_code = "en" if language == "English" else "id"
    t = texts[lang_code]  # Gunakan teks berdasarkan bahasa
    st.title(t["title"])
    st.markdown(t["description"])

    # Upload file
    uploaded_file = st.file_uploader(t["upload_label"], type=["pdf", "docx", "txt"])

    # Pilih model
    selected_model = st.sidebar.selectbox(t["model_label"], ["SVC", "Random Forest", "KNN"])

    if uploaded_file is not None:
        try:
            # Simulasi ekstraksi teks dari file (sesuaikan dengan logika Anda)
            resume_text = handle_file_upload(uploaded_file)
            st.success(t["success_message"])

            # Tampilkan teks (opsional)
            if st.checkbox(t["show_text"], False):
                st.text_area("Extracted Resume Text / Teks Resume", resume_text, height=300)

            # Prediksi
            st.subheader(t["predicted_category"])
            category = predict_category(resume_text, selected_model)
            st.write(f"{t['selected_model']}: **{selected_model}**")
            st.write(f"{t['predicted_category']}: **{category}**")

        except Exception as e:
            st.error(f"{t['error_message']}{str(e)}")


if __name__ == "__main__":
    main()
