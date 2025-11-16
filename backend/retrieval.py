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
index = pc.Index("bettercallpaul-index")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# creating vectorDB object
vectorDB = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)



def answer(query):
  # retrieving act name
  doc, score = vectorDB.similarity_search_with_relevance_scores(
      f"{query}"
  )[0]

  print(score)

  if score < 0.70:
      return "I’m not confident which law your question relates to."


  filter_data = doc.metadata["doc_title"]  # metadata based filter
  print(filter_data)

  # search for relevant chunks from documents
  relevant_chunks = vectorDB.similarity_search(
     query,
     k=10, 
     filter={"doc_title": f"{filter_data}"}
  )

  # creating langchain components

  model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
  parser = StrOutputParser()
  template = """
            You are a legal assistant specialized in Canadian federal legislation.

            You will be given:
            1. A user question about Canadian Acts or Regulations.
            2. A set of retrieved document excerpts (“context”).

            Your job:
            - Use ONLY the information found in the provided context to answer the question.
            - If the context does not contain enough information to answer confidently, say:
            “I cannot answer based on the retrieved documents.”
            - Do NOT invent facts, interpretations, penalties, definitions, or legal rules that are not explicitly stated in the context.
            - Keep the answer concise, neutral, and legally accurate.
            - If the user asks for advice, warnings, or interpretations beyond the text, respond with:
            “I can only provide information found in the retrieved documents.”

            Formatting rules:
            - Answer in 3–6 sentences unless the question requires more.
            - Cite relevant context sections using short quotes if helpful.

            Now answer the user question based on the context below.

            ---------------------
            CONTEXT:
            {context}
            ---------------------

            QUESTION:
            {question}

            ANSWER:
            """
  
  prompt = ChatPromptTemplate.from_template(template)

  chain = prompt | model | parser
  answer = chain.invoke({
      "question": f"{query}", 
      "context": "\n\n".join([chunk.page_content for chunk in relevant_chunks])
  })

#   print("\n\n".join([chunk.page_content for chunk in relevant_chunks]))
  return answer


query = "Under what circumstances can a federal institution refuse access to records on the grounds that they contain information obtained in confidence from another government??"

print(answer(query))