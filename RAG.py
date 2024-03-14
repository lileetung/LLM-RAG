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

# load first PDF
loader_data1 = PyMuPDFLoader("data/data1.pdf")
PDF_data1 = loader_data1.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
all_splits_data1 = text_splitter.split_documents(PDF_data1)

# load second PDF
loader_data2 = PyMuPDFLoader("data/data2.pdf")
PDF_data_data2 = loader_data2.load()
text_splitter_data2 = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
all_splits_data2 = text_splitter_data2.split_documents(PDF_data_data2)

# create db
persist_directory = 'db'
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}

# 初始化嵌入模型
embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# 創建並填充向量數據庫
vectordb = Chroma.from_documents(persist_directory=persist_directory)
vectordb.add_documents(documents=all_splits_data1, embedding=embedding)
vectordb.add_documents(documents=all_splits_data2, embedding=embedding)

# init Llama，download from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main
llm = LlamaCpp(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

# Prompt Template
DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> 
    You are a helpful assistant eager to assist with providing better Google search results.
    <</SYS>> 
    
    [INST] Provide an answer to the following question in 150 words. Ensure that the answer is informative, \
            relevant, and concise:
            {question} 
    [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a helpful assistant eager to assist with providing better Google search results. \
        Provide an answer to the following question in about 150 words. Ensure that the answer is informative, \
        relevant, and concise: \
        {question}""",
)

# Conditional Prompt Selector
QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)


prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)

# init retriever and QA model
retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

# chat bot
llm_chain = LLMChain(prompt=prompt, llm=llm)
print("""
      使用LLMChain進行問答
      You can use the questions below:
      1. What are the main cultural impacts of the Renaissance in Europe?
      2. What is the impact of cryptocurrency on traditional banking systems?
      """)
query = input("text the question: ")
llm_chain.invoke({"question": query})

# 使用RetrievalQA進行問答
while True:
    print("""
          使用RetrievalQA進行問答
          You can use the questions below:
          1. how much IQ does Miller have?
          2. What makes Drew Miller's artwork stand out in contemporary art?
          3. Introduce Ava Chen.
          4. How has Ava Chen's background as an engineer influenced her writing style and content?
          """)
    query = input("text the question: ")
    qa.invoke(query)
