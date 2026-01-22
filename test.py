# # Test code for document ingestion and analysis using a PDFHandler and DocumentAnalyzer

# # import os
# # from pathlib import Path
# # from src.document_analyzer.data_ingestion import DocumentHandler       # Your PDFHandler class
# # from src.document_analyzer.data_analysis import DocumentAnalyzer  # Your DocumentAnalyzer class

# # # Path to the PDF you want to test
# # PDF_PATH = r"C:\\Users\\sunny\\document_portal\\data\\document_analysis\\sample.pdf"

# # # Dummy file wrapper to simulate uploaded file (Streamlit style)
# # class DummyFile:
# #     def __init__(self, file_path):
# #         self.name = Path(file_path).name
# #         self._file_path = file_path

# #     def getbuffer(self):
# #         return open(self._file_path, "rb").read()

# # def main():
# #     try:
# #         # ---------- STEP 1: DATA INGESTION ----------
# #         print("Starting PDF ingestion...")
# #         dummy_pdf = DummyFile(PDF_PATH)

# #         handler = DocumentHandler(session_id="test_ingestion_analysis")
        
# #         saved_path = handler.save_pdf(dummy_pdf)
# #         print(f"PDF saved at: {saved_path}")

# #         text_content = handler.read_pdf(saved_path)
# #         print(f"Extracted text length: {len(text_content)} chars\n")

# #         # ---------- STEP 2: DATA ANALYSIS ----------
# #         print("Starting metadata analysis...")
# #         analyzer = DocumentAnalyzer()  # Loads LLM + parser
        
# #         analysis_result = analyzer.analyze_document(text_content)

# #         # ---------- STEP 3: DISPLAY RESULTS ----------
# #         print("\n=== METADATA ANALYSIS RESULT ===")
# #         for key, value in analysis_result.items():
# #             print(f"{key}: {value}")

# #     except Exception as e:
# #         print(f"Test failed: {e}")

# # if __name__ == "__main__":
# #     main()

# ## Testing code for document comparison using LLMs

# # import io
# # from pathlib import Path
# # from src.document_compare.data_ingestion import DocumentIngestion
# # from src.document_compare.document_comparator import DocumentComparatorLLM

# # # ---- Setup: Load local PDF files as if they were "uploaded" ---- #
# # def load_fake_uploaded_file(file_path: Path):
# #     return io.BytesIO(file_path.read_bytes())  # simulate .getbuffer()

# # # ---- Step 1: Save and combine PDFs ---- #
# # def test_compare_documents():
# #     ref_path = Path("C:\\Complete_Content2\\llmops_batch\\document_portal\\data\\document_compare\\Long_Report_V1.pdf")
# #     act_path = Path("C:\\Complete_Content2\\llmops_batch\\document_portal\\data\\document_compare\\Long_Report_V2.pdf")

# #     # Wrap them like Streamlit UploadedFile-style
# #     class FakeUpload:
# #         def __init__(self, file_path: Path):
# #             self.name = file_path.name
# #             self._buffer = file_path.read_bytes()

# #         def getbuffer(self):
# #             return self._buffer

# #     # Instantiate
# #     comparator = DocumentIngestion()
# #     ref_upload = FakeUpload(ref_path)
# #     act_upload = FakeUpload(act_path)

# #     # Save files and combine
# #     ref_file, act_file = comparator.save_uploaded_files(ref_upload, act_upload)
# #     combined_text = comparator.combine_documents()
# #     comparator.clean_old_sessions(keep_latest=3)

# #     print("\n Combined Text Preview (First 1000 chars):\n")
# #     print(combined_text[:1000])

# #     # ---- Step 2: Run LLM comparison ---- #
# #     llm_comparator = DocumentComparatorLLM()
# #     df = llm_comparator.compare_documents(combined_text)
    
# #     print("\n Comparison DataFrame:\n")
# #     print(df)

# # if __name__ == "__main__":
# #     test_compare_documents()
    
    

# # Testing code for document chat functionality

# # import sys
# # from pathlib import Path
# # from langchain_community.vectorstores import FAISS
# # from src.single_document_chat.data_ingestion import SingleDocIngestor
# # from src.single_document_chat.retrieval import ConversationalRAG
# # from utils.model_loader import ModelLoader

# # FAISS_INDEX_PATH = Path("faiss_index")

# # def test_conversational_rag_on_pdf(pdf_path:str, question:str):
# #     try:
# #         model_loader = ModelLoader()
        
# #         if FAISS_INDEX_PATH.exists():
# #             print("Loading existing FAISS index...")
# #             embeddings = model_loader.load_embeddings()
# #             vectorstore = FAISS.load_local(folder_path=str(FAISS_INDEX_PATH), embeddings=embeddings,allow_dangerous_deserialization=True)
# #             retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# #         else:
# #             # Step 2: Ingest document and create retriever
# #             print("FAISS index not found. Ingesting PDF and creating index...")
# #             with open(pdf_path, "rb") as f:
# #                 uploaded_files = [f]
# #                 ingestor = SingleDocIngestor()
# #                 retriever = ingestor.ingest_files(uploaded_files)
                
# #         print("Running Conversational RAG...")
# #         session_id = "test_conversational_rag"
# #         rag = ConversationalRAG(retriever=retriever, session_id=session_id)
# #         response = rag.invoke(question)
# #         print(f"\nQuestion: {question}\nAnswer: {response}")
                    
# #     except Exception as e:
# #         print(f"Test failed: {str(e)}")
# #         sys.exit(1)
    
# # if __name__ == "__main__":
# #     # Example PDF path and question
# #     pdf_path = "data\\single_document_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
# #     question = "What is the significance of the attention mechanism? can you explain it in simple terms?"

# #     if not Path(pdf_path).exists():
# #         print(f"PDF file does not exist at: {pdf_path}")
# #         sys.exit(1)
    
# #     # Run the test
# #     test_conversational_rag_on_pdf(pdf_path, question)
    
    
# ## testing for multidoc chat
# import sys
# from pathlib import Path
# from src.multi_document_chat.data_ingestion import DocumentIngestor
# from src.multi_document_chat.retrieval import ConversationalRAG

# def test_document_ingestion_and_rag():
#     try:
#         test_files = [
#             "data\\multi_doc_chat\\market_analysis_report.docx",
#             "data\\multi_doc_chat\\NIPS-2017-attention-is-all-you-need-Paper.pdf",
#             "data\\multi_doc_chat\\sample.pdf",
#             "data\\multi_doc_chat\\state_of_the_union.txt"
#         ]
        
#         uploaded_files = []
        
#         for file_path in test_files:
#             if Path(file_path).exists():
#                 uploaded_files.append(open(file_path, "rb"))
#             else:
#                 print(f"File does not exist: {file_path}")
                
#         if not uploaded_files:
#             print("No valid files to upload.")
#             sys.exit(1)
            
#         ingestor = DocumentIngestor()
        
#         retriever = ingestor.ingest_files(uploaded_files)
        
#         for f in uploaded_files:
#             f.close()
                
#         session_id = "test_multi_doc_chat"
        
#         rag = ConversationalRAG(session_id=session_id, retriever=retriever)
        
#         question = "what is President Zelenskyy said in their speech in parliament?"
        
#         answer=rag.invoke(question)
        
#         print("\n Question:", question)
        
#         print("Answer:", answer)
        
#         if not uploaded_files:
#             print("No valid files to upload.")
#             sys.exit(1)
            
#     except Exception as e:
#         print(f"Test failed: {str(e)}")
#         sys.exit(1)
        
# if __name__ == "__main__":
#     test_document_ingestion_and_rag()
    
    
    
# # Use this code snippet in your app.
# # If you need more information about configurations
# # or implementing the sample code, visit the AWS docs:
# # https://aws.amazon.com/developer/language/python/

# # import boto3
# # from botocore.exceptions import ClientError


# # def get_secret():

# #     secret_name = "api_keys"
# #     region_name = "ap-southeast-2"

# #     # Create a Secrets Manager client
# #     session = boto3.session.Session()
# #     client = session.client(
# #         service_name='secretsmanager',
# #         region_name=region_name
# #     )

# #     try:
# #         get_secret_value_response = client.get_secret_value(
# #             SecretId=secret_name
# #         )
# #     except ClientError as e:
# #         # For a list of exceptions thrown, see
# #         # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
# #         raise e

# #     secret = get_secret_value_response['SecretString']

# #     # Your code goes here.



# {
#   "family": "documentportaltd",
#   "networkMode": "awsvpc",
#   "executionRoleArn": "arn:aws:iam::459497895986:role/ecsTaskExecutionRole",
#   "requiresCompatibilities": ["FARGATE"],
#   "cpu": "1024",
#   "memory": "8192",
#   "containerDefinitions": [
#     {
#       "name": "document-portal-container",
#       "image": "459497895986.dkr.ecr.ap-southeast-2.amazonaws.com/documentportalliveclass",
#       "cpu": 1024,
#       "essential": true,
#       "portMappings": [
#         {
#           "containerPort": 8080,
#           "hostPort": 8080,
#           "protocol": "tcp",
#           "name": "document-portal-container-8080-tcp",
#           "appProtocol": "http"
#         }
#       ],
#       "environment": [
#         {
#           "name": "ENV",
#           "value": "production"
#         }
#       ],
#       "secrets": [
#         {
#           "name": "GROQ_API_KEY",
#           "valueFrom": "arn:aws:secretsmanager:ap-southeast-2:459497895986:secret:api_keys-nZTtj8"
#         },
        
#         {
#           "name": "GOOGLE_API_KEY",
#           "valueFrom": "arn:aws:secretsmanager:ap-southeast-2:459497895986:secret:api_keys-nZTtj8"
#         }
        
#       ],
#       "logConfiguration": {
#         "logDriver": "awslogs",
#         "options": {
#           "awslogs-group": "/ecs/documentportaltd",
#           "awslogs-region": "ap-southeast-2",
#           "awslogs-stream-prefix": "ecs",
#           "awslogs-create-group": "true"
#         }
#       }
#     }
#   ]
# }
