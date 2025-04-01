import os
from fpdf import FPDF
from gtts import gTTS
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.caches import InMemoryCache, BaseCache
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import streamlit as st
from io import BytesIO

# Set OpenCV to not use GPU if not available
os.environ["OPENCV_DONT_USE_GPU"] = "1"

# Ensure BaseCache is properly initialized before usage
BaseCache._cache = InMemoryCache()

# Ensure ChatGroq is correctly rebuilt before use
ChatGroq.model_rebuild()


# ------------------------------ App 1: Engine Inspection Class ------------------------------ #
class EngineInspectionApp:
    """Detect defects in gas turbine blades using YOLOv8 and generate reports via Groq LLM."""

    def __init__(self):
        # Initialize YOLO model for defect detection
        self.model_path = "./yolov8n_model/best.pt"
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            st.error("YOLO model weights not found. Please ensure the file exists at the specified path.")
            self.model = None  # Avoid breaking the app

        # Initialize Groq LLM (Llama model)
        try:
            self.groq_client = ChatGroq(
                model="llama3-70b-8192",
                temperature=0,
                groq_api_key=st.secrets["groq"]["api_key"],
            )
        except Exception as e:
            st.error(f"Failed to initialize Groq LLM: {str(e)}")
            self.groq_client = None

    def preprocess_image(self, image):
        """Resize image to 640x640 while maintaining aspect ratio."""
        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = image.shape[:2]
        aspect_ratio = w / h
        new_w, new_h = (640, int(640 / aspect_ratio)) if w > h else (int(640 * aspect_ratio), 640)

        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        x_offset, y_offset = (640 - new_w) // 2, (640 - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    def detect_defects(self, image):
        """Detect defects using YOLO and return annotated image."""
        if not self.model:
            return image, []

        try:
            results = self.model(image)
            annotated_image = np.copy(image)
            labels = []

            for result in results:
                for box in result.boxes:
                    cls, conf = int(box.cls[0]), float(box.conf[0])
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
        """Generate an activation report using Groq LLM."""
        if not self.groq_client:
            return "Groq LLM is not initialized."

        defect_list = ', '.join([label for label, _ in defects])
        prompt = (
            f"Generate an activation report for the following defects in gas turbine blades: {defect_list}. "
            "Include:\n1. Definitions and descriptions\n2. Maintenance procedures\n3. Inspection steps\n"
            "4. Possible causes\n5. Safety warnings."
        )
        return self.groq_client.predict(prompt)

    def run(self):
        """Streamlit application for defect detection and report generation."""
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
                    st.subheader("Activation Report")
                    st.write(report)


# ------------------------------ App 2: Interactive Chat Class ------------------------------ #
class InteractiveChatApp:
    def __init__(self):
        self.vectorstore = self.setup_vectorstore()
        self.chain = self.create_chain()

    def setup_vectorstore(self):
        """Initialize FAISS vector store."""
        try:
            index = IndexFlatL2(384)
            docstore = InMemoryDocstore({})
            index_to_docstore_id = {}
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

            return FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
        except Exception as e:
            st.error(f"Error initializing FAISS vector store: {str(e)}")
            return None

    def create_chain(self):
        """Create Conversational Retrieval Chain using Groq LLM."""
        try:
            groq_api_key = st.secrets["groq"]["api_key"]
            llm = ChatGroq(model="llama3-70b-8192", temperature=0, groq_api_key=groq_api_key)

            retriever = self.vectorstore.as_retriever() if self.vectorstore else None
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, chain_type="stuff", memory=memory)
        except Exception as e:
            st.error(f"Error creating LLM chain: {str(e)}")
            return None

    def run(self):
        """Streamlit application for interactive chat."""
        st.title("Interactive Chat with Report Export 📝")
        user_input = st.text_area("Enter your query:")

        if user_input and self.chain:
            response = self.chain.run(question=user_input)
            st.subheader("Response:")
            st.write(response)


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
