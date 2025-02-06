import streamlit as st
import cv2
import easyocr
import networkx as nx
import numpy as np
from pyvis.network import Network
from PIL import Image
import tempfile
import os

def preprocess_image(image):
    """Converts image to grayscale and applies adaptive thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def extract_text(image):
    """Uses EasyOCR to extract text from the image."""
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    return [res[1] for res in results]

def parse_relationships(extracted_text):
    """Dynamically identifies relationships using separators like '->', '=>', ':' or spaces."""
    G = nx.DiGraph()
    for line in extracted_text:
        parts = [p.strip() for p in line.replace('=>', '->').replace(':', '->').split('->')]
        if len(parts) > 1:
            for i in range(len(parts) - 1):
                G.add_edge(parts[i], parts[i + 1])
        else:
            G.add_node(parts[0])
    return G

def create_interactive_graph(G):
    """Generates an interactive flowchart using PyVis."""
    net = Network(height='600px', width='100%', directed=True)
    for node in G.nodes:
        net.add_node(node, label=node, color='#3498db')
    for edge in G.edges:
        net.add_edge(edge[0], edge[1])
    
    temp_dir = tempfile.gettempdir()
    html_path = os.path.join(temp_dir, 'flowchart.html')
    net.show(html_path)
    return html_path

def main():
    st.title("📝 Whiteboard to Flowchart Converter")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        processed_img = preprocess_image(image_np)
        extracted_text = extract_text(processed_img)
        
        st.subheader("Extracted Text")
        st.write("\n".join(extracted_text))
        
        G = parse_relationships(extracted_text)
        
        st.subheader("Generated Flow Diagram")
        html_path = create_interactive_graph(G)
        st.components.v1.html(open(html_path, 'r').read(), height=600)

if __name__ == "__main__":
    main()
