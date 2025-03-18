import os
from fpdf import FPDF
from gtts import gTTS
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import streamlit as st
from io import BytesIO
from docx import Document
import torch.serialization
from torch.nn.modules.container import Sequential  # Added for safe globals

# ------------------------------ App 1: Engine Inspection Class ------------------------------ #
class EngineInspectionApp:
    """
    EngineInspectionApp performs defect detection on gas turbine blades using YOLOv8.
    It also generates a detailed inspection report for identified defects using the Groq LLM.
    """

    def __init__(self):
        # Initialize YOLO model for defect detection
        self.model_path = 'yolov8n_model/best.pt'
        if os.path.exists(self.model_path):
            from ultralytics.nn.tasks import DetectionModel
            # Allowlist both DetectionModel and Sequential
            torch.serialization.add_safe_globals([DetectionModel, Sequential])
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                st.error(f"Failed to load YOLO model: {str(e)}")
                self.model = None
        else:
            st.error(f"YOLO model weights not found at '{self.model_path}'. Please upload or place the file.")
            self.model = None

        # Initialize Groq LLM
        try:
            self.groq_client = ChatGroq(model="llama3-70b-8192", temperature=0, groq_api_key=st.secrets["groq"]["api_key"])
        except KeyError:
            st.error("Groq API key not found in Streamlit secrets. Please configure it in secrets.toml.")
            self.groq_client = None

    def preprocess_image(self, image):
        """
        Preprocess the image to fit the YOLO model input dimensions (640x640).
        """
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)

            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            h, w = image.shape[:2]
            aspect_ratio = w / h
            if w > h:
                new_w, new_h = 640, int(640 / aspect_ratio)
            else:
                new_w, new_h = int(640 * aspect_ratio), 640

            resized = cv2.resize(image, (new_w, new_h))
            canvas = np.zeros((640, 640, 3), dtype=np.uint8)
            x_offset, y_offset = (640 - new_w) // 2, (640 - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            return canvas
        except Exception as e:
            st.error(f"Image preprocessing failed: {str(e)}")
            return None

    def detect_defects(self, image):
        """
        Detect defects using YOLO and return an annotated image and list of detected defects.
        """
        if self.model is None or image is None:
            return image, []
        
        try:
            results = self.model(image)
            annotated_image = np.copy(image)
            labels = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{self.model.names[cls]} {conf:.2f}"
                    labels.append((self.model.names[cls], conf))

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return annotated_image, labels
        except Exception as e:
            st.error(f"Error in defect detection: {str(e)}")
            return image, []

    def generate_report(self, defects):
        """
        Generate an activation report for detected defects using Groq LLM.
        """
        if self.groq_client is None:
            return "Error: Groq LLM not initialized due to missing API key."
        
        try:
            if not defects:
                return "No defects detected. No report generated."
            defect_list = ', '.join([label for label, _ in defects])
            prompt = (
                f"Generate an activation report for the following defects in gas turbine blades: {defect_list}. "
                "Include:\n1. Definitions and descriptions of each defect.\n"
                "2. Maintenance procedures.\n3. Inspection steps.\n4. Possible causes.\n"
                "5. Safety warnings and precautions."
            )
            response = self.groq_client.invoke(prompt)
            return response.content
        except Exception as e:
            st.error(f"Error in report generation: {str(e)}")
            return "Error generating the report."

    def run(self):
        """
        Streamlit application to upload an image, detect defects, and generate a comprehensive report.
        """
        st.title("Engine Component Inspection System \U0001F527")
        st.write("Upload an image to detect defects in gas turbine blades.")
        
        uploaded_file = st.file_uploader("Choose an image (jpg/png):", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            preprocessed_image = self.preprocess_image(image)
            if preprocessed_image is not None:
                annotated_image, defects = self.detect_defects(preprocessed_image)
                st.image(annotated_image, caption="Detected Defects", use_column_width=True)

                st.write("### Detected Defects:")
                if defects:
                    for label, conf in defects:
                        st.write(f"- {label} (Confidence: {conf:.2%})")
                else:
                    st.write("No defects detected.")

                if st.button("Generate Activation Report"):
                    with st.spinner("Generating report..."):
                        report = self.generate_report(defects)
                        st.subheader("Activation Report")
                        st.write(report)

# ------------------------------ App 2: Interactive Chat Class ------------------------------ #
class InteractiveChatApp:
    def __init__(self):
        self.vectorstore = self.setup_vectorstore()
        self.chain = self.create_chain()

    def setup_vectorstore(self):
        try:
            index = IndexFlatL2(384)
            docstore = InMemoryDocstore({})
            index_to_docstore_id = {}
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

            vectorstore = FAISS(embedding_function=embeddings,
                                index=index,
                                docstore=docstore,
                                index_to_docstore_id=index_to_docstore_id)
            return vectorstore
        except Exception as e:
            st.error(f"Error initializing FAISS vector store: {str(e)}")
            return None

    def create_chain(self):
        try:
            groq_api_key = st.secrets["groq"]["api_key"]
            llm = ChatGroq(model="llama3-70b-8192", temperature=0, groq_api_key=groq_api_key)
            retriever = self.vectorstore.as_retriever()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, chain_type="stuff", memory=memory)
        except Exception as e:
            st.error(f"Error creating LLM chain: {str(e)}")
            return None

    def export_response(self, response):
        """
        Export the response to audio, PDF, and DOCX formats.
        """
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("\u25B6\uFE0F Play Audio", key="audio"):
                audio_buffer = BytesIO()
                gTTS(response).save(audio_buffer)
                audio_buffer.seek(0)
                st.audio(audio_buffer, format="audio/mp3")

        with col2:
            if st.button("\U0001F4C4 Generate PDF", key="pdf"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, response)
                pdf_buffer = BytesIO()
                pdf.output(pdf_buffer)
                pdf_buffer.seek(0)
                st.download_button("Download PDF", pdf_buffer, file_name="response.pdf", mime="application/pdf")

        with col3:
            if st.button("\U0001F4C4 Generate DOCX", key="docx"):
                doc = Document()
                doc.add_paragraph(response)
                doc_buffer = BytesIO()
                doc.save(doc_buffer)
                doc_buffer.seek(0)
                st.download_button("Download DOCX", doc_buffer, file_name="response.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    def run(self):
        """
        Streamlit application for interactive chat and comprehensive report generation.
        """
        st.title("Interactive Chat with Report Export \U0001F4DC")
        st.write("Ask a question or provide input for a response.")
        
        user_input = st.text_area("Enter your query:", height=100)
        if user_input and st.button("Submit Query"):
            if self.chain:
                with st.spinner("Processing..."):
                    response = self.chain.run(question=user_input)
                    st.subheader("Response:")
                    st.write(response)
                    self.export_response(response)
            else:
                st.error("Chat chain not initialized properly. Check setup.")

# ------------------------------ Main Entry Point ------------------------------ #
if __name__ == "__main__":
    st.set_page_config(page_title="Defect Detection and LLM Chat App", layout="wide")

    st.sidebar.title("Application Selection")
    app_selection = st.sidebar.selectbox("Choose an Application:", ["Engine Inspection", "Interactive Chat"])

    if app_selection == "Engine Inspection":
        engine_app = EngineInspectionApp()
        engine_app.run()
    elif app_selection == "Interactive Chat":
        chat_app = InteractiveChatApp()
        chat_app.run()
