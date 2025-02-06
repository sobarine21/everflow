import streamlit as st
import cv2
import easyocr
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

def extract_text(image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    text = "\n".join([res[1] for res in results])
    return text

def create_graph(extracted_text):
    G = nx.DiGraph()
    lines = extracted_text.strip().split("\n")
    nodes = []
    edges = []
    for line in lines:
        parts = line.split("->")
        for part in parts:
            if part.strip() not in nodes:
                nodes.append(part.strip())
        if len(parts) == 2:
            edges.append((parts[0].strip(), parts[1].strip()))
    
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def draw_graph(G):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
    st.pyplot(plt)

st.title("Whiteboard to Flowchart Converter")
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    processed_img = preprocess_image(image)
    extracted_text = extract_text(processed_img)
    
    st.subheader("Extracted Text")
    st.write(extracted_text)
    
    G = create_graph(extracted_text)
    
    st.subheader("Generated Flow Diagram")
    draw_graph(G)
