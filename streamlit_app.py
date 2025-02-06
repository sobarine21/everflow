import streamlit as st
import cv2
import easyocr
import networkx as nx
import numpy as np
from pyvis.network import Network
from PIL import Image
import tempfile
import os

# Function to preprocess the image
def preprocess_image(image):
    """Enhances the image for better text and shape recognition."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged

# Function to extract both text and shapes
def extract_text_and_shapes(image):
    """Extracts both text and basic shapes from the image."""
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    text_data = [res[1] for res in results]
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_data = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]
    
    return text_data, shape_data

# Function to parse the relationships dynamically
def parse_relationships(text_data, shape_data):
    """Dynamically identifies relationships from text and spatial positions."""
    G = nx.DiGraph()
    
    elements = text_data + [f"Shape_{i}" for i in range(len(shape_data))]
    for i in range(len(elements) - 1):
        G.add_edge(elements[i], elements[i + 1])
    
    return G

# Function to create an interactive graph using pyvis
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

# Main function to tie everything together
def main():
    st.title("📝 Handwritten Notes to Visual Flowchart")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Read and display the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess the image
        processed_img = preprocess_image(image_np)
        
        # Extract text and shapes from the image
        text_data, shape_data = extract_text_and_shapes(processed_img)
        
        # Display extracted data
        st.subheader("Extracted Elements")
        st.write("Text Data:", text_data)
        st.write("Detected Shapes:", shape_data)
        
        # Parse relationships and generate the graph
        G = parse_relationships(text_data, shape_data)
        
        # Create an interactive graph
        st.subheader("Generated Flow Diagram")
        html_path = create_interactive_graph(G)
        
        # Display the generated graph
        st.components.v1.html(open(html_path, 'r').read(), height=600)

# Ensure this runs when the script is executed directly
if __name__ == "__main__":
    main()
