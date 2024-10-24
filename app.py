import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

import streamlit as st
import pandas as pd
import uuid
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
import requests

# Configure your LLM
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Streamlit UI
st.title("Job Posting Email Generator")

# Input for Job Posting URL
job_url = st.text_input("Enter Job Posting URL")

# Input for Portfolio/CV
portfolio_cv = st.file_uploader("Upload your Portfolio/CV", type=["csv"])

# Button to generate email
if st.button("Generate Email"):
    if job_url and portfolio_cv:
        # Load job description
        loader = WebBaseLoader(job_url)
        try:
            page_data = loader.load().pop().page_content
            st.success("Scraped job data successfully.")
        except Exception as e:
            st.error(f"Error scraping the job description: {e}")
            page_data = ""

        if page_data:
            # Extract job details
            prompt_extract = PromptTemplate.from_template(
                """
                ### SCRAPED TEXT FROM WEBSITE:
                {page_data}
                ### INSTRUCTION:
                The scraped text is from the career's page of a website.
                Your job is to extract the job postings and return them in JSON format containing the 
                following keys: `role`, `experience`, `skills` and `description`.
                Only return the valid JSON.
                ### VALID JSON (NO PREAMBLE):    
                """
            )

            chain_extract = prompt_extract | llm 
            res = chain_extract.invoke(input={'page_data': page_data})
            json_parser = JsonOutputParser()
            json_res = json_parser.parse(res.content)

            # Display the extracted job details in Streamlit
            st.write("Extracted Job Details:")
            if json_res:
                job_details = json_res[0]  # Assuming json_res is a list of job details
                st.write("Role:", job_details['role'])
                st.write("Experience:", job_details['experience'])
                st.write("Skills:")
                for skill in job_details['skills']:
                    st.write("-", skill)  # Display each skill on a new line
                st.write("Description:", job_details['description'])
                
                # Generate email
                df = pd.read_csv(portfolio_cv)
                client = chromadb.PersistentClient('vectorstore')
                collection = client.get_or_create_collection(name="portfolio")

                if not collection.count():
                    for _, row in df.iterrows():
                        collection.add(documents=row["Techstack"],
                                       metadatas={"links": row["Links"]},
                                       ids=[str(uuid.uuid4())])

                links = collection.query(query_texts=["experience in python","experience in react"], n_results=2).get('metadatas', [])
                job = json_res
                prompt_email = PromptTemplate.from_template(
                    """
                    ### JOB DESCRIPTION:
                    {job_description}

                    ### INSTRUCTION:
                    You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
                    the seamless integration of business processes through automated tools. 
                    Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
                    process optimization, cost reduction, and heightened overall efficiency. 
                    Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
                    in fulfilling their needs.
                    Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
                    Remember you are Mohan, BDE at AtliQ. 
                    Do not provide a preamble.
                    ### EMAIL (NO PREAMBLE):
                    """
                )

                chain_email = prompt_email | llm
                res = chain_email.invoke({"job_description": str(job), "link_list": links})
                st.write("Generated Email:")
                st.text(res.content)
            else:
                st.error("Failed to extract job details.")
    else:
        st.warning("Please enter the job posting URL and upload your Portfolio/CV.")
