# show_suggester.py
import csv
import pickle
from thefuzz import process
import numpy as np
import requests
import time
from dotenv import load_dotenv
import os
from PIL import Image
from io import BytesIO
import openai
import re
from annoy import AnnoyIndex

# Load environment variables from .env file
load_dotenv()
# Get the API key from the environment
api_key_lightx = os.getenv("LIGHTX_API_KEY")
if not api_key_lightx:
    print("Error: Missing API key.")
api_key_openai = os.getenv("OPENAI_API_KEY")
if not api_key_openai:
    print("Error: Missing API key.")
    

def load_embeddings(filename):
    try:
        with open(filename, "rb") as f:
            embeddings_dict = pickle.load(f)
        return embeddings_dict
    except FileNotFoundError:
        print(
            f"Pickle file '{filename}' not found. Please run the embedding script first."
        )
        exit(1)


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


# Function to match the user's input to the real show names using fuzzy matching
def fuzzy_match(show_name, tv_shows):
    names = [show["name"] for show in tv_shows]
    best_match = process.extractOne(show_name, names)
    return best_match


def get_tv_shows_input():
    while True:
        user_input = input(
            "Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show: "
        )
        shows = [show.strip() for show in user_input.split(",")]
        if len(shows) > 1:
            return shows
        else:
            print("Please enter more than one TV show.")


def confirm_tv_shows(shows):
    while True:
        confirm = input(f"Making sure, do you mean {', '.join(shows)}? (y/n): ").lower()
        if confirm == "y":
            return True
        elif confirm == "n":
            print(
                "Sorry about that. Let's try again, please make sure to write the names of the tv shows correctly."
            )
            return False
        else:
            print("Invalid input. Please answer with 'y' for yes or 'n' for no.")
            return False


def generate_recommendations(user_shows, tv_shows, embeddings_dict, num_trees=10):
    # Normalize the embeddings dictionary by converting keys to lowercase
    normalized_embeddings_dict = {
        show_name.lower(): embedding for show_name, embedding in embeddings_dict.items()
    }

    # Step 1: Build an Annoy index for all TV show embeddings
    dim = len(
        next(iter(normalized_embeddings_dict.values()))
    )  

    # Create an Annoy index
    annoy_index = AnnoyIndex(dim, "angular")

    # Add embeddings to the Annoy index
    for i, show in enumerate(tv_shows):
        show_name = show["name"].lower()  # Normalize the TV show name to lowercase
        show_embedding = normalized_embeddings_dict.get(show_name)
        if show_embedding is not None:
            annoy_index.add_item(i, show_embedding)

    # Build the index
    annoy_index.build(num_trees) 
    # Step 2: Get embeddings for user liked shows and calculate the average vector
    user_embeddings = []
    for user_show in user_shows:
        # Normalize user input by converting to lowercase
        normalized_user_show = user_show.strip().lower()
        matched_show, fuzzy_score = process.extractOne(
            normalized_user_show, [show["name"].lower() for show in tv_shows]
        )

        embedding = normalized_embeddings_dict.get(matched_show)
        if embedding is not None:
            user_embeddings.append(embedding)
        else:
            print(f"Embedding for {matched_show} not found.")

    # Step 3: Calculate the average embedding
    if user_embeddings:
        average_vector = np.mean(user_embeddings, axis=0)
    else:
        return []

    # Step 4: Find the nearest neighbors using Annoy
    nearest_neighbors = annoy_index.get_nns_by_vector(
        average_vector, 5, include_distances=True
    )

    # Step 5: Get the recommended shows
    recommendations = []
    for idx, dist in zip(nearest_neighbors[0], nearest_neighbors[1]):
        show = tv_shows[idx]

        similarity_score = 1 - dist
        similarity_score = max(0, min(similarity_score, 1))

        # Convert to percentage
        similarity_percentage = round(similarity_score * 100, 3)
        recommendations.append((show["name"], similarity_percentage))

    # Step 6: Filter out user input shows from recommendations
    recommendations = [
        (show_name, score)
        for show_name, score in recommendations
        if show_name.lower() not in [show.lower() for show in user_shows]
    ]

    # Step 7: Ensure exactly 5 recommendations by adding more if needed
    if len(recommendations) < 5:
        # Add more recommendations from the remaining shows that aren't already in the user input
        remaining_shows = [
            (show["name"], score)
            for show, score in zip(tv_shows, nearest_neighbors[0])
            if show["name"].lower()
            not in [show_name.lower() for show_name, _ in recommendations]
        ]
        # Add the remaining recommendations until we have 5
        recommendations.extend(remaining_shows[: 5 - len(recommendations)])
        recommendations.sort(key=lambda x: x[1], reverse=True)

    # Return exactly 5 recommendations
    return recommendations[:5]


def generate_image_with_lightx(show_name, show_description, api_key):
    url = 'https://api.lightxeditor.com/external/api/v1/text2image'
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key
    }

    # Create the data payload with both the show name and description as the text prompt
    data = {
        "textPrompt": f"An image inspired by the TV show '{show_name}'and the Description: {show_description}"
    }

    # Send request to generate image
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        # Retrieve the orderId for status checking
        order_id = response.json()['body']['orderId']
        # Now check the status of the image generation
        check_url = 'https://api.lightxeditor.com/external/api/v1/order-status'
        status_payload = {
            "orderId": order_id
        }

        retries = 0
        max_retries = 5
        status = "init"
        image_url = None

        # Keep checking the status until the image is ready or retries are exhausted
        while status != "active" and retries < max_retries:
            status_response = requests.post(check_url, headers=headers, json=status_payload)

            if status_response.status_code == 200:
                status_info = status_response.json()['body']
                status = status_info['status']
                if status == "active":
                    image_url = status_info['output']
                    break
            else:
                print(f"Failed to check status. Status code: {status_response.status_code}")
                break

            # Wait for 3 seconds before checking again
            time.sleep(3)
            retries += 1

        if image_url:
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                img = Image.open(BytesIO(image_response.content))
                return img
        else:
            print("Image generation failed or was not completed in time.")
            return None
    else:
        return None


def shows_creator(recommendations, user_input, api_key_openai):
    # Prepare OpenAI API key
    openai.api_key = api_key_openai

    # Step 1: Create a prompt for generating a new TV show based on user input
    user_input_prompt = f"""Generate a new TV show based on the fact that the user loved these shows: {', '.join(user_input)}.
    The new show should be original and exciting, with a unique storyline and a name that fits the genre of the input shows.
    Return the output as a plain text in this format: Show name:<showname>, Description (not longer than 2 lines MAX):START WITH it is about...'"""

    # Step 2: Create the conversation list for OpenAI API
    conversation1 = [
        {"role": "system", "content": "You are a helpful, proffesional tv shows creator and assistant."},
        {"role": "user", "content": user_input_prompt},
    ]

    # Step 3: Generate the custom show #1 based on user input using ChatCompletion
    response1 = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Use the gpt-4o-mini model
        messages=conversation1,
        temperature=0.7,
        max_tokens=150,
    )

    # Step 4: Create a prompt for generating a new TV show based on recommended shows
    recommended_shows_prompt = f"""Generate a new TV show based on the recommendation: {recommendations[0][0]}. 
    Make it interesting, creative, and related to the genre of the recommended show.
    Return the output as a plain text in this format: Show name:<showname>, Description (not longer than 2 lines MAX):START WITH it is about...'"""

    # Step 5: Create the conversation list for OpenAI API
    conversation2 = [
        {
            "role": "system",
            "content": "You are a helpful, proffesional tv shows creator and assistant.",
        },
        {"role": "user", "content": recommended_shows_prompt},
    ]

    # Step 6: Generate the custom show #2 based on recommended shows using ChatCompletion
    response2 = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Use the gpt-4o-mini model
        messages=conversation2,
        temperature=0.7,
        max_tokens=150,
    )

    # Step 7: Extract the show names and descriptions from OpenAI's response
    def parse_show_response(response):
        # Extract the show text from the response
        show_text = response["choices"][0]["message"]["content"].strip()

        # Define regular expressions to extract show name and description
        name_pattern = r'Show name:\s*(.*?)\s*(?=Description:)'
        description_pattern = r'Description: (.*?)(?=\n|$)'
        show_name = re.search(name_pattern, show_text)
        show_description = re.search(description_pattern, show_text)

        # Extract the values if they exist, otherwise set to None
        show_name = show_name.group(1) if show_name else None
        show_description = show_description.group(1) if show_description else None

        return show_name, show_description
    show1_name, show1_description = parse_show_response(response1)
    show2_name, show2_description = parse_show_response(response2)

    # Step 8: Return the custom shows for display
    return (
        f"I have also created just for you two shows which I think you would love.\n\n"
        f"Show #1 is based on the fact that you loved the input shows that you gave me. Its name is {show1_name} and {show1_description}.\n\n"
        f"Show #2 is based on the shows that I recommended for you. Its name is {show2_name} and {show2_description}.\n\n"
        f"Here are also the 2 TV show ads. Hope you like them!"
    )


# Cosine similarity function
def cosine_similarity(a, b):
    a = np.squeeze(a)
    b = np.squeeze(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    tv_shows = load_tv_shows("imdb_tvshows.csv")  # Load the TV shows data from CSV
    embeddings_dict = load_embeddings(
        "tv_shows_embeddings.pkl"
    )  # Load the embeddings from the pickle file

    # Step 1: Ask for user input (TV shows they like)
    user_shows = get_tv_shows_input()

    # Step 2: Confirm the user's input
    while not confirm_tv_shows(user_shows):
        user_shows = get_tv_shows_input()

    # Step 3: Generate recommendations based on user input
    print("Great! Generating recommendations now...")
    recommendations = generate_recommendations(user_shows, tv_shows, embeddings_dict)

    # Display the recommendations
    print("Here are the TV shows that I think you would love:")
    for recommendation in recommendations:
        print(f"{recommendation[0]} ({recommendation[1]}%)")

    # Step 4: Generate and display custom show suggestions
    custom_show_message = shows_creator(recommendations,user_shows, api_key_openai)
    print(custom_show_message)
    # Generate images for the top 2 recommendations
    image1 = generate_image_with_lightx(
        recommendations[0][0], tv_shows[0]["description"], api_key_lightx
    )
    image2 = generate_image_with_lightx(
        recommendations[1][0], tv_shows[1]["description"], api_key_lightx
    )

    # Show the images
    if image1:
        image1.show()
    if image2:
        image2.show()


if __name__ == "__main__":
    main()