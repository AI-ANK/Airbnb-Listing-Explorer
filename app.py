# Required imports
import streamlit as st
import pandas as pd
from llama_index.core import SimpleDirectoryReader, ServiceContext, StorageContext, VectorStoreIndex
import re
from llama_index.core.vector_stores import (
            MetadataFilter,
            MetadataFilters,
            FilterOperator,
            FilterCondition
        )

from llama_index.readers.file import PagedCSVReader

from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq

from llama_index.embeddings.fastembed import FastEmbedEmbedding
from qdrant_client import QdrantClient
import json
import os
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase, ServiceContext
from llama_index.core.query_engine import NLSQLTableQueryEngine
from pathlib import Path
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.query_engine import SQLAutoVectorQueryEngine, RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.retrievers import VectorIndexAutoRetriever

from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine import RetrieverQueryEngine
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(layout="wide")
write_dir = Path("airbnbtextdata")

# Initialize Qdrant client
client = QdrantClient(
    url=os.environ['QDRANT_URL'], 
    api_key=os.environ['QDRANT_API_KEY'],
)

# Initialize LLM and embedding model
# llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
llm = Groq(model="mixtral-8x7b-32768", api_key=os.getenv('GROQ_API'))


embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(chunk_size_limit=1024, llm=llm, embed_model=embed_model)

vector_store = QdrantVectorStore(client=client, collection_name="airbnb_5")
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# Function to extract and format text data from a dataframe row
def get_text_data(data):
    return f"""
    Review Date: {data['review_date']}
    Reviewer Name: {data['Reviewer Name']}
    Comments: {data['Comments']}
    Listing URL: {data['Listing URL']}
    Listing Name: {data['Listing Name']}
    Host URL: {data['Host URL']}
    Host Name: {data['Host Name']}
    Host Since: {data['Host Since']}
    Host Response Time: {data['Host Response Time']}
    Host Response Rate: {data['Host Response Rate']}
    Is Superhost: {data['Is Superhost']}
    Neighbourhood: {data['neighbourhood']}
    Neighborhood Group: {data['Neighborhood Group']}
    City: {data['City']}
    Postal Code: {data['Postal Code']}
    Country Code: {data['Country Code']}
    Latitude: {data['Latitude']}
    Longitude: {data['Longitude']}
    Is Exact Location: {data['Is Exact Location']}
    Property Type: {data['Property Type']}
    Room Type: {data['Room Type']}
    Accommodates: {data['Accomodates']}
    Bathrooms: {data['Bathrooms']}
    Bedrooms: {data['Bedrooms']}
    Beds: {data['Beds']}
    Square Feet: {data['Square Feet']}
    Price: {data['Price']}
    Guests Included: {data['Guests Included']}
    Min Nights: {data['Min Nights']}
    Reviews: {data['Reviews']}
    First Review: {data['First Review']}
    Last Review: {data['Last Review']}
    Overall Rating: {data['Overall Rating']}
    Accuracy Rating: {data['Accuracy Rating']}
    Cleanliness Rating: {data['Cleanliness Rating']}
    Checkin Rating: {data['Checkin Rating']}
    Communication Rating: {data['Communication Rating']}
    Location Rating: {data['Location Rating']}
    Value Rating: {data['Value Rating']}
    """

@st.cache_data
def load_data():
    # if write_dir.exists():
    reader = PagedCSVReader(encoding="utf-8")
    documents = reader.load_data(file=Path("Airbnb Berlin 1000.csv"))
    return documents

documents = load_data()


# loader2 = PagedCSVReader(encoding="utf-8")
# documents2 = loader2.load_data(file=Path("First_100_Rows.csv"))

for doc in documents:
    # Regular expression to extract key-value pairs
    pattern = re.compile(r'(\w+[\s\w]*?):\s*(.*)')

    # Creating a dictionary to store the parsed data
    parsed_data = {match.group(1).strip(): match.group(2).strip() for match in pattern.finditer(doc.text)}
    # st.write(parsed_data['Latitude'])
    # Setting the parsed data as metadata

     # Check if 'Latitude' and 'Longitude' are available in parsed data to avoid KeyError
    if 'Latitude' in parsed_data and 'Longitude' in parsed_data:
        # Creating a nested 'location' dictionary
        location = {
            "lon": float(parsed_data['Longitude']),
            "lat": float(parsed_data['Latitude'])
        }
        parsed_data['location'] = location
    doc.metadata = parsed_data

    # Optionally print the metadata to confirm it's set correctly
    # print(doc.metadata)
# st.write("HERE")
# st.write(documents)

#Create vector indexes and store in Qdrant. To be run only once in the beginning
# index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, service_context=service_context, storage_context=storage_context, show_progress=True)

# Load the vector index from Qdrant collection
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context, embed_model=embed_model
)

# Streamlit UI setup
st.title('Airbnb Listing Explorer - Berlin')

# Load the dataset
df_file_path = 'Airbnb Berlin 1000.csv'  # Path to the csv file
if os.path.exists(df_file_path):
    df = pd.read_csv(df_file_path)
    df["text"] = df.apply(get_text_data, axis=1)
    st.dataframe(df)  # Display df in the UI
else:
    st.error("Data file not found. Please check the path and ensure it's correct.")

# Input from user
user_query = st.text_input("Enter your query:", "Find a good property in Friedrichshain with Overall rating above 96")

# # Define the options for the radio button
# options = ['Simple: Qdrant Similarity Search + LLM Call (works well for filtering type of queries)', 'Advanced: Qdrant Similarity Search + Llamaindex Text-to-SQL']

# # Create a radio button for the options
# selection = st.radio("Choose an option:", options)

# Processing the query
if st.button("Submit Query"):
    # Execute different blocks of code based on the selection
    # if selection == 'Simple: Qdrant Similarity Search + LLM Call (works well for filtering type of queries)':
        # Part 1, semantic search + LLM call
    # Generate query vector
    query_vector = embed_model.get_query_embedding(user_query)
    # Perform search with Qdrant

    # Regular expression pattern to match the date format MM-DD-YY
    date_pattern = re.compile(r'\b(\d{2}-\d{2}-\d{2})\b')

    # Search for a date in the user query
    date_match = date_pattern.search(user_query)

    # Initialize filters list
    base_filters = []

    # Check if a date was found and add a date filter
    if date_match:
        date_value = date_match.group(0)  # Extract the matched date
        base_filters.append(
            MetadataFilter(key="review_date", operator=FilterOperator.EQ, value=date_value)
        )

    # Create neighborhood mapping
    unique_neighborhoods = df[['neighbourhood', 'Latitude', 'Longitude']].drop_duplicates(subset='neighbourhood')
    neighborhood_mapping = unique_neighborhoods.set_index('neighbourhood').to_dict(orient='index')

    selected_neighborhood = None
    # Check for neighborhood in user query
    for neighborhood in neighborhood_mapping:
        if neighborhood in user_query:
            selected_neighborhood = neighborhood
            break

    if selected_neighborhood:  # Check if neighborhood was found in the query
        lat = str(neighborhood_mapping[selected_neighborhood]["Latitude"])
        lon = str(neighborhood_mapping[selected_neighborhood]["Longitude"])

        # Add location filters
        base_filters.append(
            MetadataFilter(key="Latitude", operator=FilterOperator.TEXT_MATCH, value=lat)
        )
        base_filters.append(
            MetadataFilter(key="Longitude", operator=FilterOperator.CONTAINS, value=lon)
        )

    # Check if filters were added and combine them under a MetadataFilters object with an AND condition
    if base_filters:
        filters = MetadataFilters(
            filters=base_filters,
            condition=FilterCondition.AND,
        )
        # st.write(filters)
    else:
        st.write("No valid filters applied based on the user query.")
        filters = []
    # filters = MetadataFilters(
    #     filters=[
    #         MetadataFilter(key="Latitude", operator=FilterOperator.TEXT_MATCH, value='52.53752'),
    #         MetadataFilter(key="Longitude", operator=FilterOperator.CONTAINS, value="13.42168"),
    #     ],
    #     condition=FilterCondition.AND,
    # )

    retriever = index.as_retriever(filters=filters)
    response = retriever.retrieve(user_query)
    # st.write (response)
    
    # response = client.search(collection_name="airbnb_2", query_vector=query_vector, limit=10)
    # st.write(response)
    # Processing and displaying the results
    text = ''
    properties_list = []  # List to store multiple property dictionaries
    for scored_point in response:
        # print(scored_point.text)
        # Access the payload, then parse the '_node_content' JSON string to get the 'text'
        # node_content = json.loads(scored_point.payload['_node_content'])
        text += f"\n{scored_point.text}\n"    
        # Initialize a new dictionary for the current property
        property_dict = {}
        last_key = None  # Track the last key for appending text

        for line in scored_point.text.split('\n'):
            # st.write(line)
            if ':' in line:  # Check if there is a colon in the line
                key, value = line.split(': ', 1)
                property_dict[key.strip()] = value.strip()
                last_key = key.strip()  # Update last_key with the current key
            elif last_key:  # Handle the case where no colon is found and last_key is defined
                # Append the current line to the last key's value, adding a space for separation
                property_dict[last_key] += ' ' + line.strip()

        # Add the current property dictionary to the list
        properties_list.append(property_dict)

    # st.write(properties_list)
    # properties_list contains all the retrieved property dictionaries
    with st.status("Retrieving points/nodes based on user query", expanded = True) as status:
        for property_dict in properties_list:
            st.json(json.dumps(property_dict, indent=4))
            # print(property_dict)
        status.update(label="Retrieved points/nodes based on user query", state="complete", expanded=False)
    
    with st.status("Generating response based on Similarity Search + LLM Call", expanded = True) as status:
        prompt_template = f"""
            Using the below context information respond to the user query.
            context: '{properties_list}'
            query: '{user_query}'
            Response structure should look like this:

            *Detailed Response*
            
            *Relevant Details in wellformatted Markdown Table Format. Select appropriate columns based on user query*

            """
        llm_response = llm.complete(prompt_template)
        response_parts = llm_response.text.split('```')
        st.markdown(response_parts[0])
