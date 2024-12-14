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
# Load environment variables from .env file
load_dotenv()
# Get the API key from the environment
api_key = os.getenv("LIGHTX_API_KEY")
if not api_key:
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


def generate_recommendations(user_shows, tv_shows, embeddings_dict):
    # Step 1: Get embeddings for user liked shows
    user_embeddings = []
    for user_show in user_shows:
        matched_show, fuzzy_score = process.extractOne(
            user_show, [show["name"] for show in tv_shows]
        )

        embedding = embeddings_dict.get(matched_show)
        if embedding is not None:
            user_embeddings.append(embedding)
        else:
            print(f"Embedding for {matched_show} not found.")

    # Step 2: Calculate the average embedding
    if user_embeddings:
        average_vector = np.mean(user_embeddings, axis=0)
    else:
        return []

    # Step 3: Find the 5 closest shows
    recommendations = []
    for show in tv_shows:
        show_embedding = embeddings_dict.get(show["name"])
        if show_embedding is not None:
            similarity_score = cosine_similarity(average_vector, show_embedding)
            recommendations.append((show["name"], similarity_score))

    # Sort by similarity score (higher is better)
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Filter out user input shows
    recommendations = [rec for rec in recommendations if rec[0] not in user_shows]

    # Limit to top 5 recommendations
    recommendations = recommendations[:5]

    # Step 4: Calculate percentage similarity for each recommended show
    if recommendations:
        min_sim = recommendations[-1][1]
        max_sim = recommendations[0][1]
        for i in range(len(recommendations)):
            rec_name, rec_score = recommendations[i]
            percentage = (
                ((rec_score - min_sim) / (max_sim - min_sim)) * 100
                if max_sim != min_sim
                else 100
            )
            recommendations[i] = (rec_name, round(percentage, 3))

    # Return recommendations
    return recommendations


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
        print(f"Request was successful! Order ID: {order_id}")

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
            print(f"Image generated successfully! Image URL: {image_url}")
            # You can use the image URL to download or display the image
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                img = Image.open(BytesIO(image_response.content))
                return img
            else:
                print(f"Failed to retrieve the image from URL. Status code: {image_response.status_code}")
        else:
            print("Image generation failed or was not completed in time.")
            return None
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None


def shows_creator(recommendations, tv_shows):



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
    custom_show1, custom_show2 = shows_creator(
        recommendations, tv_shows
    )
    print(custom_show1)
    print(custom_show2)
    print("Here are also the 2 TV show ads. Hope you like them!")
    # Generate image for the top 2 recommendations
    image1 = generate_image_with_lightx(recommendations[0][0], api_key)
    image2 = generate_image_with_lightx(recommendations[1][0], api_key)

    # Show the images
    if image1:
        image1.show()
    if image2:
        image2.show()


if __name__ == "__main__":
    main()