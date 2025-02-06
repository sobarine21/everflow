import streamlit as st
import easyocr
import graphviz
from PIL import Image
import numpy as np
import io

def extract_text_from_image(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(np.array(image), detail=0)
    return result

def generate_flowchart(extracted_text):
    graph = graphviz.Digraph(format='png')
    previous_node = None
    
    for i, text in enumerate(extracted_text):
        node_name = f'Node{i}'
        graph.node(node_name, text)
        if previous_node:
            graph.edge(previous_node, node_name)
        previous_node = node_name
    
    return graph

# Streamlit UI
st.title("Handwritten to Flowchart Generator")
st.write("Upload a whiteboard or handwritten flow diagram, and we will generate a structured flowchart for you.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Extracting text from the image...")
    extracted_text = extract_text_from_image(image)
    
    if extracted_text:
        st.write("Extracted Text:")
        st.write(extracted_text)
        
        st.write("Generating Flowchart...")
        flowchart = generate_flowchart(extracted_text)
        flowchart.render("flowchart", format='png')
        st.image("flowchart.png", caption='Generated Flowchart', use_column_width=True)
        
        with open("flowchart.png", "rb") as file:
            btn = st.download_button(
                label="Download Flowchart",
                data=file,
                file_name="flowchart.png",
                mime="image/png"
            )
    else:
        st.write("No text detected. Try another image.")
