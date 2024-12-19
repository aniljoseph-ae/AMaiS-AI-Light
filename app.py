import os
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import streamlit as st

from langchain_community.vectorstores import Chroma, FAISS
# from langchain_core.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory

from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # Updated import path

# Rest of the code remains the same, just updating the import paths
class EngineInspectionApp:
    def __init__(self):
        self.model_path = os.getenv('YOLO_MODEL_PATH', 'yolov8n_model/best.pt')
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
        
        try:
            self.groq_client = ChatGroq(
                model="llama3-70b-8192", 
                temperature=0, 
                groq_api_key=st.secrets["groq"]["api_key"]
            )
        except Exception as e:
            st.error(f"Error initializing Groq client: {str(e)}")
            return

    # Rest of the EngineInspectionApp class remains the same
    def preprocess_image(self, image):
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
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
            st.error(f"Error preprocessing image: {str(e)}")
            return None

    def detect_defects(self, image):
        try:
            results = self.model(image)
            annotated_image = np.copy(image)
            labels = []
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{self.model.names[cls]} {conf:.2f}"
                    labels.append((self.model.names[cls], conf))
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return annotated_image, labels
        except Exception as e:
            st.error(f"Error detecting defects: {str(e)}")
            return None, []

    def generate_report(self, defects):
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
            st.error(f"Error generating report: {str(e)}")
            return None

    def run(self):
        st.title("Engine Component Inspection System 🛠️")
        uploaded_file = st.file_uploader("Upload an image (jpg/png):", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                preprocessed_image = self.preprocess_image(image)
                if preprocessed_image is not None:
                    annotated_image, defects = self.detect_defects(preprocessed_image)
                    if annotated_image is not None:
                        st.image(annotated_image, caption="Detected Defects", use_column_width=True)
                        st.write("### Detected Defects:")
                        for label, conf in defects:
                            st.write(f"- {label} (Confidence: {conf:.2%})")
                        
                        if st.button("Generate Activation Report"):
                            with st.spinner("Generating report..."):
                                report = self.generate_report(defects)
                                if report:
                                    st.subheader("Activation Report")
                                    st.write(report)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

class InteractiveChatApp:
    def __init__(self):
        try:
            self.vectorstore_chroma = self.setup_chroma_vectorstore()
            self.vectorstore_faiss = self.setup_faiss_vectorstore()
            self.chain_chroma = self.create_chain(self.vectorstore_chroma)
            self.chain_faiss = self.create_chain(self.vectorstore_faiss)
        except Exception as e:
            st.error(f"Error initializing chat app: {str(e)}")

    @staticmethod
    @st.cache_resource
    def setup_chroma_vectorstore():
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
            return Chroma(persist_directory="chatLLM_vector_store", embedding_function=embeddings)
        except Exception as e:
            st.error(f"Error setting up Chroma vectorstore: {str(e)}")
            return None

    @staticmethod
    @st.cache_resource
    def setup_faiss_vectorstore():
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
            return FAISS.load_local("visionLLM_vector_store", embeddings)
        except Exception as e:
            st.error(f"Error setting up FAISS vectorstore: {str(e)}")
            return None

    def create_chain(self, vectorstore):
        try:
            if vectorstore is None:
                raise ValueError("Vectorstore is not initialized")
            
            groq_api_key = st.secrets["groq"]["api_key"]
            llm = ChatGroq(model="llama3-70b-8192", temperature=0, groq_api_key=groq_api_key)
            retriever = vectorstore.as_retriever()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                memory=memory
            )
        except Exception as e:
            st.error(f"Error creating chain: {str(e)}")
            return None

    def run(self):
        st.title("Interactive Chat with Vector Stores 📜")
        mode = st.radio("Select Vector Store:", ["Chroma", "FAISS"])
        chain = self.chain_chroma if mode == "Chroma" else self.chain_faiss

        if chain is None:
            st.error("Chat system is not properly initialized")
            return

        user_input = st.text_area("Enter your query:")
        if user_input:
            try:
                response = chain.run(question=user_input)
                st.subheader("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

def main():
    st.set_page_config(page_title="Defect Detection and Chat App", layout="wide")
    
    if 'app_selection' not in st.session_state:
        st.session_state['app_selection'] = "Engine Inspection"

    st.session_state['app_selection'] = st.sidebar.selectbox(
        "Choose an Application:",
        ["Engine Inspection", "Interactive Chat"],
        index=["Engine Inspection", "Interactive Chat"].index(st.session_state['app_selection'])
    )

    try:
        if st.session_state['app_selection'] == "Engine Inspection":
            engine_app = EngineInspectionApp()
            engine_app.run()
        else:
            chat_app = InteractiveChatApp()
            chat_app.run()
    except Exception as e:
        st.error(f"Error running application: {str(e)}")

if __name__ == "__main__":
    main()
