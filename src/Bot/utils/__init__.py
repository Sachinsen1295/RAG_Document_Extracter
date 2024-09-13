# import ocrmypdf
# from src.Bot.logger import logging


# class OCR:
#     def __init__(self,input, output):
#         self.input = input
#         self.output = output

#     def do_ocr(self):
#         ocrmypdf.ocr(self.input, output_file=self.output)
#         return self.output



import ocrmypdf
from src.Bot.logger import logging
import os

class OCR:
    def __init__(self, input, output=None):
        self.input = input
        # Set default output path if none is provided
        if output is None:
            default_output_dir = os.path.join(os.getcwd(), "output")  # Default directory for output files
            os.makedirs(default_output_dir, exist_ok=True)  # Create the directory if it doesn't exist
            self.output = os.path.join(default_output_dir, "output.pdf")  # Default output file path
        else:
            self.output = output

    def do_ocr(self):
        ocrmypdf.ocr(self.input, output_file=self.output,force_ocr=True,)
        return self.output


    # Function to reset the FAISS index (clear vectors)
    def reset_faiss_index(vector_store):
        """Clear all vectors from the FAISS index."""
        if isinstance(vector_store.index, faiss.Index):
            vector_store.index.reset()
            print("FAISS index has been reset (vectors cleared).")
        else:
            print("No FAISS index found.")

    # Function to delete the FAISS index (remove from memory)
    def delete_faiss_index(vector_store):
        """Delete the FAISS index and free up memory."""
        if isinstance(vector_store.index, faiss.Index):
            del vector_store.index
            vector_store.index = None  # Set to None to avoid further access
            gc.collect()  # Ensure memory is freed
            print("FAISS index deleted and memory cleared.")
        else:
            print("No FAISS index found.")




