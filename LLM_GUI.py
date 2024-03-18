import tkinter as tk
from tkinter import scrolledtext
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

class LLM_GUI:
    def __init__(self, master):
        self.master = master
        master.title("LLM Interaction")

        # GUI Setup
        self.display_text = scrolledtext.ScrolledText(master, height=10, bg="black")
        self.display_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.display_text.tag_configure('user', foreground='white')
        self.display_text.tag_configure('ai', foreground='lightgreen')
        self.display_text.config(state=tk.DISABLED)

        self.user_input = tk.Entry(master)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), pady=(0, 10))

        self.send_button = tk.Button(master, text="↑", command=self.send_query)
        self.send_button.pack(side=tk.LEFT, padx=(0, 10), pady=(0, 10))

        # LangChain Initialization
        self.init_langchain()

    def init_langchain(self):
        # Document Processing and Embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

        # PDF1
        loader1 = PyMuPDFLoader("data/data1.pdf")  
        document_text1 = loader1.load()
        document_splits1 = text_splitter.split_documents(document_text1)

        # PDF2
        loader2 = PyMuPDFLoader("data/data2.pdf")
        document_text2 = loader2.load()
        document_splits2 = text_splitter.split_documents(document_text2)
        
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        
        # Vector Store
        self.vectordb = Chroma.from_documents(documents=document_splits1, embedding=embedding, persist_directory='db')
        self.vectordb.add_documents(documents=document_splits2, embedding=embedding)
        
        # LLM Initialization
        self.llm = LlamaCpp(
            model_path="llama-2-7b-chat.Q4_K_M.gguf",  # Example path, replace with actual path
            n_gpu_layers=100,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
        )
        
        prompt_template = PromptTemplate(input_variables=["question"], template="Your question: {question}")
        self.llm_chain = LLMChain(prompt=prompt_template, llm=self.llm)

    def upload_data(self):
        pass

    def send_query(self):
        query = self.user_input.get()
        if query:
            self.insert_text(f"USER: {query}", 'user')
            self.user_input.delete(0, tk.END)
            self.insert_text("Model is processing...", 'ai')
            self.master.after(1000, self.model_reply, query)

    def model_reply(self, query):
        # Simulate retrieving response from the model
        # For actual model invocation, replace this with actual code to retrieve the response
        retriever = self.vectordb.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=retriever, 
            verbose=True)
        model_output = qa.invoke(query)  # Assuming this returns a dictionary

        # Ensure the output has a 'result' key
        if 'result' in model_output:
            response = model_output['result']
        else:
            response = "Error: Model output does not contain a 'result' key."
        
        # Remove "Model is processing..."
        self.remove_processing_text()
        
        # Display the model response
        self.insert_text(f"AI: {response}", 'ai')
        self.insert_text(f"---", 'user')

    def remove_processing_text(self):
        """Remove the 'Model is processing...' text from the display."""
        processing_text_index = self.display_text.search("Model is processing...", "1.0", tk.END)
        if processing_text_index:
            self.display_text.config(state=tk.NORMAL)
            line_end_index = f"{processing_text_index} lineend + 1c"
            self.display_text.delete(processing_text_index, line_end_index)
            self.display_text.config(state=tk.DISABLED)

    def insert_text(self, text, tag):
        """Insert text into the display text widget with proper line breaks."""
        self.display_text.config(state=tk.NORMAL)
        
        # Check if the widget is not empty and add a newline character if needed
        if self.display_text.get("1.0", tk.END).strip():
            self.display_text.insert(tk.END, "\n")
        
        self.display_text.insert(tk.END, text, tag)
        self.display_text.see(tk.END)
        self.display_text.config(state=tk.DISABLED)



def main():
    root = tk.Tk()
    app = LLM_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
