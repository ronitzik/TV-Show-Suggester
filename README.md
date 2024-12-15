# ShowSuggesterAI

**ShowSuggesterAI** is a Python program that provides personalized TV show recommendations based on a user's favorite TV shows. The program leverages machine learning techniques such as fuzzy string matching, embeddings, vector search, and AI-generated images to generate accurate recommendations and unique TV show ads.

## Technologies Used

- **Rapidfuzz**: Used for fuzzy string matching to ensure accurate mapping of the user's input show names to the correct titles in the database.
- **Prompt_toolkit**: Provides an interactive command-line interface to enhance the user experience with auto-completion, syntax highlighting, and better handling of user input.
- **Embeddings (OpenAI API)**: Used to generate vector representations (embeddings) for TV show descriptions. This helps in computing similarity between user preferences and available shows.
- **Lightx Image Generator API**: Generates AI-generated ads for TV shows based on user preferences and recommendations. 
- **Vector Search (Annoy/Usearch)**: Used for fast and efficient nearest neighbor searches for similar TV show embeddings. This allows for quick retrieval of the most relevant TV shows from a large list.

## How It Works

### 1. User Input
The user is prompted to provide a list of TV shows they enjoyed watching. They are asked to type show names separated by commas. The program then confirms if the show names are correctly recognized.

### 2. Fuzzy Matching
The user-provided TV shows are matched with the official show names in the database using fuzzy string matching (Levenshtein distance) through the **Rapidfuzz** library. This ensures that slight typos or variations in spelling do not hinder the program's ability to match shows correctly.

### 3. Embeddings and Vector Search
Each show’s description is processed to create an embedding using the **OpenAI API**. These embeddings are stored and loaded from a pickle file to avoid repeated API calls. An average embedding vector is generated from the user’s input shows. This average vector is then compared with the embeddings of the available shows using **vector search** methods (Annoy), allowing the program to find the most similar TV shows based on the user's preferences.

### 4. Recommendations
The program ranks the closest matching shows and provides the top 5 recommendations, including a similarity score expressed as a percentage.

### 5. Show Ads
In addition to the recommendations, the program generates two custom show ads using the **Lightx Image Generator API**. The ads are based on the user's favorite shows and the recommendations generated, providing a visual representation of the suggested content.