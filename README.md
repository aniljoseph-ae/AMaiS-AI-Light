
# **Aircraft Maintenance System AI (AMaiS AI)**

## **Overview**
AMaiS AI is a comprehensive, AI-driven application designed to assist in aircraft maintenance. By leveraging state-of-the-art technologies such as YOLO for defect detection and advanced large language models (LLMs) for detailed reporting, this system aims to enhance inspection accuracy, streamline report generation, and provide interactive query support.

## **Key Features**
1. **Engine Inspection System**  
   - Detect defects in gas turbine blades using the YOLOv8 model.
   - Generate detailed aerospace engineering quality reports with insights on defect causes, corrective actions, inspection procedures, and cautions using LLMs (Groq-powered Llama model).

2. **Interactive Chat System**  
   - Respond to user queries with AI-driven insights.
   - Use FAISS for efficient vector storage and retrieval.
   - Export responses in multiple formats (PDF, DOCX, Audio).

3. **User-Friendly Interface**  
   - Intuitive Streamlit-based GUI.
   - Background customization for branding.

## **Technologies Used**
- **Computer Vision**: YOLOv8 for defect detection.
- **Natural Language Processing**: 
  - LLMs (Groq's Llama3-70b) for report generation and conversational AI.
  - HuggingFace embeddings for semantic similarity in vector storage.
- **Data Storage**: FAISS for efficient vector indexing and retrieval.
- **UI Framework**: Streamlit for an interactive, user-friendly interface.
- **Utilities**: 
  - FPDF for PDF generation.
  - gTTS for text-to-speech audio output.
  - Python libraries such as PIL, OpenCV, and NumPy for image processing.

## **Installation**

### Prerequisites
- Python 3.8 or later
- pip (Python package manager)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/amai-ai.git
   cd amai-ai
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your API keys for the Groq LLM in the Streamlit secrets file:
   ```plaintext
   .streamlit/secrets.toml
   ```

4. Ensure that your YOLO model weights are in the correct path:
   ```plaintext
   models/vision_model/mediumModel_yolov8m_3590Img.pt
   ```

5. Place your custom background images and icons in the `static` directory.

## **Usage**
Run the application using the following command:
```bash
streamlit run app.py
```

### Navigation
- **Engine Inspection**: Upload an image of a turbine blade to detect defects and generate a detailed report.
- **Interactive Chat**: Enter queries related to aircraft maintenance and receive AI-powered responses.

### Outputs
- Annotated images highlighting defects.
- Detailed defect reports in professional aerospace engineering format.
- Downloadable response formats (PDF, DOCX, Audio).

## **Key Functionalities**

### Engine Inspection System
- **Image Preprocessing**: Resizes and centers images for optimal YOLO input.
- **Defect Detection**: Identifies defects with bounding boxes and confidence scores.
- **Report Generation**: Uses LLMs to generate in-depth reports with actionable insights.

### Interactive Chat System
- **Semantic Search**: Retrieves relevant context using FAISS and HuggingFace embeddings.
- **Conversational AI**: Provides meaningful responses based on user queries.
- **Response Export**: Offers audio, PDF, and DOCX formats for user convenience.

## **Future Enhancements**
- Integration with cloud storage for report archiving.
- Real-time defect detection using live camera feeds.
- Multi-language support for reports and responses.
- Incorporation of advanced transformer-based models like Vision Transformers and BERT for enhanced accuracy.

## **Acknowledgments**
This project leverages cutting-edge open-source tools and libraries. Special thanks to:
- Ultralytics for YOLO
- HuggingFace for NLP models
- FAISS for vector storage
- Streamlit for the application framework

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

## **Contact**
For questions, feedback, or contributions, please contact:
- **Anil Joseph**  
- Email: [aniljoseph.ae@gmail.com]  
- GitHub: [github.com/aniljoseph](https://github.com/aniljoseph-ae)  

---
