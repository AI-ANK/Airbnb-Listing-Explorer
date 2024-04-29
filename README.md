# AI Real Estate Search - Powered by Qdrant and Llamaindex

his tool transforms how users search for houses, moving beyond traditional keyword-based engines. By leveraging advanced capabilities of Qdrant and LlamaIndex, it uses a combination of Vector Similarity Search, LLM and Text-to-SQL to searching for properties using natural human language.

## Features
- **Interactive UI**: Explore New York's housing market through a user-friendly web interface.
- **Natural Language Queries**: Use simple or complex natural language queries to find properties.
- **Dynamic Search Approaches**: Choose between simple similarity searches or advanced text-to-SQL queries for detailed inquiries.
- **Embedding Generation**: Automatically convert property details into embeddings using Qdrant's FastEmbed library, stored in Qdrant Cloud.
- **Real-Time Results**: Instantly retrieve and explore the top property matches based on your query.

## How to Use
- **Set Up Your Environment**: Ensure Python and Streamlit are installed. Clone the repository and install dependencies.
- **Configure Qdrant**: Before launching the application, ensure you have set up a Qdrant cluster. This involves creating a cluster in the Qdrant Cloud and noting down the URL and API key. You'll need these to configure your application to communicate with the Qdrant database. See this link for more details: https://qdrant.tech/documentation/cloud/quickstart-cloud/
- **Prepare the Qdrant Collection**: Determine the name for your Qdrant collection where the property embeddings will be stored. If you're running the application for the first time, ensure the code on line 52-54 of app.py to create and populate the collection is uncommented and correctly configured with your chosen collection name.
- **Launch the Application**: Execute `streamlit run app.py` to start the app.
- **Search for Properties**: Enter your housing query, select a search approach, and view the matching properties.
- **Explore Results**: Interact with the search results to find more about each property.


## Installation
```git clone https://github.com/AI-ANK/AI-Real-Estate-Search-Qdrant.git```

```cd AI-Real-Estate-Search-Qdrant```

```pip install -r requirements.txt```

Create a .env file in the root directory of the project and add your OpenAI API key, Qdrant URL, and Qdrant API key. Replace the placeholder values with your actual keys:
```
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

## Tools and Technologies
- **UI**: Streamlit
- **Vector Store**: Qdrant
- **Vector Embeddings**: Qdrant's FastEmbed
- **LLM Orchestration and Text-to-SQL**: Llamaindex
- **LLM**: OpenAI GPT-3.5

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support
For support, please open an issue in the GitHub issue tracker.

## Live Demo
Try the demo [here](https://huggingface.co/spaces/AI-ANK/AI-Real-Estate-Search-Qdrant-Llamaindex)!

## Authors
### Developed by [Harshad Suryawanshi](https://www.linkedin.com/in/harshadsuryawanshi/)
If you find this project useful, consider giving it a ⭐ on GitHub!
