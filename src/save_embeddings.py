# save_embeddings.py
import openai
import pickle
import csv
import os
from dotenv import load_dotenv

load_dotenv()  
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002", 
        input=text,
    )
    return response["data"][0]["embedding"]


def load_tv_shows(csv_file):
    tv_shows = []
    with open(csv_file, "r", encoding="UTF-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            tv_shows.append(
                {
                    "name": row["Title"],
                    "genre": row["Genres"],
                    "description": row["Description"],
                }
            )
    return tv_shows


def save_embeddings(tv_shows, filename):
    embeddings_dict = {}
    for show in tv_shows:
        print(f"Fetching embedding for {show['name']}...")
        embedding = get_embedding(show["description"])
        embeddings_dict[show["name"]] = embedding

    with open(filename, "wb") as f:
        pickle.dump(embeddings_dict, f)
    print(f"Embeddings saved to {filename}")


def main():
    tv_shows = load_tv_shows("imdb_tvshows.csv")
    save_embeddings(tv_shows, "tv_shows_embeddings.pkl")


if __name__ == "__main__":
    main()
