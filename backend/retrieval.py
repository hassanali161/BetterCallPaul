import os
from dotenv import load_dotenv

# langchain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone


# api keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# args for vectorDB object
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("bettercallpaul-descriptions-index")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# creating vectorDB object
vectorDB = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)



def answer(query):
  # retrieving documents to filter through
  results = vectorDB.similarity_search_with_relevance_scores(
      f"{query}",
      k = 5
  )

  # print(results)

  # valid documents to scrape and input in documents index
  valid_documents = []

  for i in range(len(results)):
     print(results[i][1])
     if results[i][1] >= 0.70:
        valid_documents.append({
           "name": results[i][0].metadata["doc_title"],
           "link": results[i][0].metadata["doc_link"]
        })

  # if score < 0.70:
  #     return "I’m not confident which law your question relates to."


  # name = doc.metadata["doc_title"]  # metadata to retrieve name of document
  # link = doc.metadata["doc_link"]  # metadata to retrieve reference link for document

  # # search for relevant chunks from documents
  # relevant_chunks = vectorDB.similarity_search(
  #    query,
  #    k=10, 
  #    filter={"doc_title": f"{name}"}
  # )

  # # creating langchain components

  # model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
  # parser = StrOutputParser()
  # template = """
  #           You are a legal assistant specialized in Canadian federal legislation.

  #           You will be given:
  #           1. A user question about Canadian Acts or Regulations.
  #           2. A set of retrieved document excerpts (“context”).
  #           3. A document NAME and LINK for the reference section.

  #           Your strict instructions:
  #           - You MUST answer ONLY using information found in the provided CONTEXT.
  #           - If the context does not contain enough information to answer confidently, say:
  #             "I cannot answer based on the retrieved documents."
  #           - Do NOT invent legal definitions, rules, penalties, commentary, or interpretations
  #             that are not explicitly stated in the context.
  #           - If the user asks for legal advice, warnings, or interpretations beyond the text, say:
  #             "I can only provide information found in the retrieved documents."
  #           - Keep the answer concise (3–6 sentences), neutral, and purely informational.

  #           ### REQUIRED OUTPUT FORMAT
  #           You MUST produce BOTH sections below.  
  #           Do not omit, rename, or reorder them.

  #           Example Output Format:
  #           ANSWER:
  #           <your 3–6 sentence answer here>

  #           REFERENCE:
  #           Document Name: <document name>
  #           Link: <document link>

  #           ### Now answer using ONLY the information below.

  #           ---------------------
  #           CONTEXT:
  #           {context}
  #           ---------------------

  #           QUESTION:
  #           {question}

  #           NAME:
  #           {name}

  #           LINK:
  #           {link}

  #           ### Write your output now following the REQUIRED OUTPUT FORMAT strictly.
  #           ANSWER:
  #           """
  
  # prompt = ChatPromptTemplate.from_template(template)

  # chain = prompt | model | parser
  # answer = chain.invoke({
  #     "question": f"{query}", 
  #     "context": "\n\n".join([chunk.page_content for chunk in relevant_chunks]),
  #     "name": f"{name}",
  #     "link": f"{link}"
  # })

  # return answer

  return valid_documents


query = "How was the zoning of the Victoria international airport done?"

print("Paul is thinking...")
print(answer(query))