import streamlit as st
from PyPDF2 import PdfFileReader, PdfFileWriter
from pdfrw import PdfReader, PdfWriter
import fitz
from pdf2image import convert_from_bytes
from io import  BytesIO 
from PIL import Image


def pdf_to_images(pdf_file):
    return convert_from_bytes(pdf_file.read())

def images_to_pdf(image_files):
    image_objects = [Image.open(image).convert('RGB') for image in image_files]
    output = BytesIO()
    # Save all images in the list to a PDF file
    image_objects[0].save(output, save_all=True, append_images=image_objects[1:], format='PDF')
    output.seek(0)  # Move to the beginning of the BytesIO buffer
    return output

def pdf_to_thumbnail(pdf_file):
    # Load PDF
    doc = fitz.open("pdf", pdf_file.getvalue())
    page = doc.load_page(0)  # Get the first page
    pix = page.get_pixmap()
    img = Image.open(BytesIO(pix.tobytes("ppm")))
    img.resize((150,150))
    
    return img

# Function to compress PDF
def compress_pdf(input_pdf, output_pdf, compression_level):
    doc = fitz.open(stream=input_pdf.read(), filetype="pdf")

    for page in doc:
        # This cleans up the content of the page, and rewrites the contents more efficiently
        page.clean_contents()

    # Save the PDF with deflation compression to reduce size
    doc.save(output_pdf, garbage=4, deflate=True, clean=True)
    doc.close()

# Function to combine two PDFs
def combine_pdfs(pdf1, pdf2, output_pdf):
    reader1 = PdfReader(pdf1)
    reader2 = PdfReader(pdf2)
    writer = PdfWriter(output_pdf)
    writer.addpages(reader1.pages)
    writer.addpages(reader2.pages)
    writer.write()

# Function to remove a page from PDF
def remove_page(input_pdf, page_number, output_pdf):
    reader = PdfReader(input_pdf)
    writer = PdfWriter(output_pdf)
    for i in range(len(reader.pages)):
        if i != page_number:
            writer.addpage(reader.pages[i])
    writer.write()

# Function to add a page to PDF
def add_page(input_pdf, page_to_add, position, output_pdf):
    reader = PdfReader(input_pdf)
    writer = PdfWriter(output_pdf)
    for i in range(len(reader.pages)):
        if i == position:
            writer.addpage(page_to_add)
        writer.addpage(reader.pages[i])
    writer.write()

def reorder_pages(input_pdf, page_order, output_pdf):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    
    # Add pages in the specified new order
    for index in page_order:
        writer.add_page(reader.pages[index])
    
    with open(output_pdf, "wb") as f:
        writer.write(f)

# Function to lock PDF
def lock_pdf(input_pdf, password, output_pdf):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    # Add all pages to the writer
    for page in reader.pages:
        writer.add_page(page)
    
    # Encrypt PDF
    writer.encrypt(password)
    
    with open(output_pdf, "wb") as f:
        writer.write(f)

# Function to unlock PDF
def unlock_pdf(input_pdf, password, output_pdf):
    reader = PdfReader(input_pdf)
    if reader.is_encrypted:
        reader.decrypt(password)
    
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    
    with open(output_pdf, "wb") as f:
        writer.write(f)

def pdf_to_thumbnails(pdf_file, num_pages=10):
    """Generate thumbnails for the first 'num_pages' of the PDF."""
    doc = fitz.open("pdf", pdf_file.getvalue())
    thumbnails = []
    # Generate thumbnail for the first 'num_pages' pages or total pages if fewer
    for page_number in range(min(num_pages, doc.page_count)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.open(BytesIO(pix.tobytes("ppm")))
        # Resize image to reduce memory usage
        img.resize((150,150))# Resize to 100x100 pixels
        thumbnails.append(img)
    return thumbnails


def main():
    st.title("PDF Editor")

    st.sidebar.title("Choose PDF operation")

    operations = ["Compress PDF", "Combine PDFs", "Remove Page", "Add Page", "Reorder Pages", "Lock PDF", "Unlock PDF","Convert Images to PDF"]
    choice = st.sidebar.selectbox("Operation", operations)

    if choice == "Compress PDF":
        st.header("Compress PDF")
        pdf_file = st.file_uploader("Upload PDF", type="pdf")
        compression_level = st.selectbox("Select Compression Level", ["low", "good", "extreme"], index=1)

        if pdf_file:
            output = BytesIO()
            compress_pdf(pdf_file, output, compression_level)
            output.seek(0)  # Important: move back to the start of the BytesIO object before downloading
            st.download_button("Download Compressed PDF", data=output, file_name="compressed.pdf", mime="application/pdf")

    elif choice == "Combine PDFs":
        st.header("Combine PDFs")
        pdf_file1 = st.file_uploader("Upload PDF 1", type="pdf")
        pdf_file2 = st.file_uploader("Upload PDF 2", type="pdf")
        if pdf_file1 and pdf_file2:
            thumbnails=[pdf_to_thumbnail(pdf_file1),pdf_to_thumbnail(pdf_file2)]
            cols_per_row = 4
            rows = [st.columns(cols_per_row) for _ in range((len(thumbnails) + cols_per_row - 1) // cols_per_row)]
            
            # Iterate over thumbnails and put them into the grid
            for idx, thumbnail in enumerate(thumbnails):
                col = rows[idx // cols_per_row][idx % cols_per_row]
                with col:
                    st.image(thumbnail, caption=f'Page {idx + 1}')
            
            output = BytesIO()
            combine_pdfs(pdf_file1, pdf_file2, output)
            st.download_button("Download Combined PDF", output, file_name="combined.pdf")

    elif choice == "Remove Page":
        st.header("Remove Page from PDF")
        pdf_file = st.file_uploader("Upload PDF", type="pdf")
        page_number = st.number_input("Page number to remove", min_value=0, value=0)
        if pdf_file:
            thumbnails = pdf_to_thumbnails(pdf_file)  # Adjust 'num_pages' as needed
        # Define how many columns you want in your grid
            cols_per_row = 4
            rows = [st.columns(cols_per_row) for _ in range((len(thumbnails) + cols_per_row - 1) // cols_per_row)]
            
            # Iterate over thumbnails and put them into the grid
            for idx, thumbnail in enumerate(thumbnails):
                col = rows[idx // cols_per_row][idx % cols_per_row]
                with col:
                    st.image(thumbnail, caption=f'Page {idx + 1}')
            output = BytesIO()
            remove_page(pdf_file, page_number, output)
            st.download_button("Download PDF", output, file_name="modified.pdf")

    elif choice == "Add Page":
        st.header("Add Page to PDF")
        pdf_file = st.file_uploader("Upload PDF", type="pdf")
        page_file = st.file_uploader("Upload Page to Add (PDF)", type="pdf")
        position = st.number_input("Position to add the page", min_value=0, value=0)
        if pdf_file and page_file:
            output = BytesIO()
            page_reader = PdfReader(page_file)
            page_to_add = page_reader.pages[0]
            add_page(pdf_file, page_to_add, position, output)
            st.download_button("Download PDF", output, file_name="modified.pdf")
    elif choice == "Reorder Pages":
        st.header("Reorder PDF Pages")
        pdf_file = st.file_uploader("Upload PDF", type="pdf")
        page_order = st.text_input("Enter new page order (comma-separated indices):")
        
        if pdf_file :

            if page_order:
                page_order = list(map(int, page_order.split(',')))
            thumbnails = pdf_to_thumbnails(pdf_file)  # Adjust 'num_pages' as needed
        # Define how many columns you want in your grid
            cols_per_row = 4
            rows = [st.columns(cols_per_row) for _ in range((len(thumbnails) + cols_per_row - 1) // cols_per_row)]
            
            # Iterate over thumbnails and put them into the grid
            for idx, thumbnail in enumerate(thumbnails):
                col = rows[idx // cols_per_row][idx % cols_per_row]
                with col:
                    st.image(thumbnail, caption=f'Page {idx + 1}')
            if st.button("Reorder Pages"):
                output = BytesIO()
                reorder_pages(pdf_file, page_order, output)
                st.download_button("Download Reordered PDF", output, file_name="reordered.pdf")

    elif choice == "Lock PDF":
        st.header("Lock PDF")
        pdf_file = st.file_uploader("Upload PDF", type="pdf")
        password = st.text_input("Enter Password for PDF", type="password")
        if pdf_file and password:
            if st.button("Lock PDF"):
                output = BytesIO()
                lock_pdf(pdf_file, password, output)
                st.download_button("Download Locked PDF", output, file_name="locked.pdf")

    elif choice == "Unlock PDF":
        st.header("Unlock PDF")
        pdf_file = st.file_uploader("Upload PDF", type="pdf")
        password = st.text_input("Enter Password for PDF", type="password")
        if pdf_file and password:
            if st.button("Unlock PDF"):
                output = BytesIO()
                unlock_pdf(pdf_file, password, output)
                st.download_button("Download Unlocked PDF", output, file_name="unlocked.pdf")

    elif choice == "Convert Images to PDF":
        st.header("Convert Images to PDF")
        # Allow users to upload multiple images
        uploaded_files = st.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

        if uploaded_files:
            cols_per_row = 4
            rows = [st.columns(cols_per_row) for _ in range((len(uploaded_files) + cols_per_row - 1) // cols_per_row)]
            
            # Iterate over thumbnails and put them into the grid
            for idx, image in enumerate(uploaded_files):
                col = rows[idx // cols_per_row][idx % cols_per_row]
                with col:
                    st.image(image, caption=f'Img {idx + 1}')
            
            # Convert images to PDF on button click
            if st.button("Convert to PDF"):
                pdf_output = images_to_pdf(uploaded_files)
                st.download_button(label="Download PDF", data=pdf_output, file_name="converted.pdf", mime="application/pdf")
                st.success("Conversion successful!")            
        


main()