# Airbnb Listing Explorer - Powered by Qdrant and Llamaindex

This tool transforms how users search for Airbnb listings, extending beyond traditional filters to include natural language queries. By harnessing the capabilities of Qdrant for vector search and Llamaindex for orchestrating Large Language Models (LLMs), it provides a nuanced search experience that interprets user queries semantically.

## Features
- Interactive UI: A Streamlit-based web interface that is easy to use.
- Natural Language Queries: Users can search using natural language, enhancing the flexibility and intuitiveness of the search process.
- Semantic Search Capabilities: Employs semantic understanding to deliver results that closely match user queries.
- Vector Storage: Utilizes Qdrant to store and retrieve vector embeddings, ensuring efficient and scalable search capabilities.
- Real-Time Results: Delivers immediate search results, offering users a dynamic interaction with the application.

## How to Use
- **Set Up Your Environment**: Ensure Python and Streamlit are installed. Clone the repository and install dependencies.
- **Configure Qdrant**: Before launching the application, ensure you have set up a Qdrant cluster. This involves creating a cluster in the Qdrant Cloud and noting down the URL and API key. You'll need these to configure your application to communicate with the Qdrant database. See this link for more details: https://qdrant.tech/documentation/cloud/quickstart-cloud/
- **Prepare the Qdrant Collection**: Determine the name for your Qdrant collection where the property embeddings will be stored. If you're running the application for the first time, ensure to run the code on line 141 of app.py to create and populate the collection.
- **Launch the Application**: Execute `streamlit run app.py` to start the app.
- **Explore Listings**: Input your search queries in natural language and view the relevant Airbnb listings.



## Installation
```git clone https://github.com/AI-ANK/Airbnb-Listing-Explorer.git```

```cd Airbnb-Listing-Explorer```

```pip install -r requirements.txt```

Create a .env file in the root directory of the project and add your OpenAI API key, Qdrant URL, and Qdrant API key. Replace the placeholder values with your actual keys:
```
GROQ_API=your_groq_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

## Tools and Technologies
- **UI**: Streamlit
- **Vector Store**: Qdrant
- **Vector Embeddings**: Qdrant's FastEmbed
- **LLM Orchestration and Text-to-SQL**: Llamaindex
- **LLM**: Mixtral 7bx8 via Groq API

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support
For support, please open an issue in the GitHub issue tracker.

## Authors
### Developed by [Harshad Suryawanshi](https://www.linkedin.com/in/harshadsuryawanshi/)
If you find this project useful, consider giving it a ⭐ on GitHub!
