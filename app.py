import os
from fpdf import FPDF
from gtts import gTTS
import numpy as np
import pickle
import cv2
from PIL import Image
from ultralytics import YOLO
#------------------------------------------------#
import torch
from ultralytics.nn.tasks import DetectionModel
#------------------------------------------------#
from langchain_community.vectorstores import FAISS
# from langchain.docstore import InMemoryDocstore
from langchain_community.docstore.in_memory import InMemoryDocstore

from faiss import IndexFlatL2
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import streamlit as st
from io import BytesIO
# NEW MODIFICATION FOR TEST
import os
import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO
import streamlit as st
from langchain_groq import ChatGroq

#---------------MODIFICATION : TESTING-----------------------------------------#
class EngineInspectionApp:
    """
    EngineInspectionApp performs defect detection on gas turbine blades using YOLOv8.
    It also generates a detailed inspection report for identified defects using the Groq LLM.
    """

    def __init__(self):
        """
        Initialize YOLO model for defect detection.
        """
        self.model_path = os.path.abspath("./yolov8n_model/best.pt")  # Path to YOLO model weights

        if os.path.exists(self.model_path):
            try:
                # Load YOLO model using the Ultralytics library
                self.model = YOLO(self.model_path)
            except Exception as e:
                st.error(f"Error loading YOLO model: {str(e)}")
                self.model = None
        else:
            st.error(f"Model file not found at {self.model_path}")
            self.model = None

        # Initialize Groq LLM (Llama model)
        self.groq_client = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            groq_api_key=st.secrets["groq"]["api_key"]
        )

    def preprocess_image(self, uploaded_file):
        """Preprocess image: Convert to RGB if needed and resize."""
        image = Image.open(uploaded_file)

        # ✅ Convert image to RGB (remove alpha channel if present)
        image = image.convert("RGB")

        image = np.array(image)  # Convert to NumPy array
        h, w = image.shape[:2]
        aspect_ratio = w / h

        if w > h:
            new_w, new_h = 640, int(640 / aspect_ratio)
        else:
            new_w, new_h = int(640 * aspect_ratio), 640

        # ✅ Ensure image is correctly resized
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return image

    def detect_defects(self, image):
        """
        Detect defects using YOLO and return an annotated image and list of detected defects.
        """
        if self.model is None:
            st.error("YOLO model not loaded.")
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
                    cv2.putText(annotated_image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return annotated_image, labels
        except Exception as e:
            st.error(f"Error in defect detection: {str(e)}")
            return image, []

    def generate_report(self, defects):
        """
        Generate an activation report for detected defects using Groq LLM.
        """
        if not defects:
            return "No defects detected."

        try:
            defect_list = ', '.join([label for label, _ in defects])
            prompt = (
                f"Generate an activation report for the following defects in gas turbine blades: {defect_list}. "
                "Include:\n1. Definitions and descriptions of each defect.\n"
                "2. Maintenance procedures.\n3. Inspection steps.\n4. Possible causes.\n"
                "5. Safety warnings and precautions."
            )
            return self.groq_client.predict(prompt)
        except Exception as e:
            st.error(f"Error in report generation: {str(e)}")
            return "Error generating the report."

    def run(self):
        """
        Streamlit application to upload an image, detect defects, and generate a comprehensive report.
        """
        st.title("Engine Component Inspection System 🔧")
        uploaded_file = st.file_uploader("Upload an image (jpg/png):", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            preprocessed_image = self.preprocess_image(image)
            annotated_image, defects = self.detect_defects(preprocessed_image)

            st.image(annotated_image, caption="Detected Defects", use_column_width=True)
            st.write("### Detected Defects:")
            for label, conf in defects:
                st.write(f"- {label} (Confidence: {conf:.2%})")

            if st.button("Generate Activation Report"):
                with st.spinner("Generating report..."):
                    report = self.generate_report(defects)
                    st.write(report)

#----------------------------------------------------------------------------------------------#


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
            if st.button("\u25B6\uFE0F Play Audio"):
                audio_path = "response.mp3"
                gTTS(response).save(audio_path)
                st.audio(audio_path)

        with col2:
            if st.button("\U0001F4C4 Generate PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, response)
                pdf.output("response.pdf")
                st.download_button("Download PDF", "response.pdf")

        with col3:
            if st.button("\U0001F4C4 Generate DOCX"):
                from docx import Document
                doc = Document()
                doc.add_paragraph(response)
                doc.save("response.docx")
                st.download_button("Download DOCX", "response.docx")

    def run(self):
        """
        Streamlit application for interactive chat and comprehensive report generation.
        """
        st.title("Interactive Chat with Report Export \U0001F4DC")
        user_input = st.text_area("Enter your query:")
        if user_input:
            response = self.chain.run(question=user_input)
            st.subheader("Response:")
            st.write(response)
            self.export_response(response)


# ------------------------------ Main Entry Point ------------------------------ #
if __name__ == "__main__":
    st.set_page_config(page_title="Defect Detection and LLM Chat App", layout="wide")

    app_selection = st.sidebar.selectbox("Choose an Application:", ["Engine Inspection", "Interactive Chat"])

    if app_selection == "Engine Inspection":
        engine_app = EngineInspectionApp()
        engine_app.run()
    elif app_selection == "Interactive Chat":
        chat_app = InteractiveChatApp()
        chat_app.run()
