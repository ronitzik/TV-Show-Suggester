import csv
import pickle
from thefuzz import process


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
    recommendations = []
    for user_show in user_shows:
        # Find the closest matching show name in the real shows list using fuzzy matching
        matched_show, fuzzy_score = process.extractOne(
            user_show, [show["name"] for show in tv_shows]
        )

        # Get the embedding for the matched show
        embedding = embeddings_dict.get(matched_show)
        if embedding is not None:   
            similarity_score = fuzzy_score
            recommendations.append((matched_show, similarity_score))
        else:
            print(f"Embedding for {matched_show} not found.")

    # Sort recommendations based on similarity score
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:3]
    # Ensure we return at least 1 recommendation
    if not recommendations:
        print("Sorry, no recommendations found. Please try again with different shows.")
    return recommendations


def generate_custom_shows(shows, recommendations, tv_shows):
    if not recommendations:
        return (
            "Sorry, no recommendations found. Please try again.",
            "Sorry, no recommendations found. Please try again.",
        )
    else:
        # Get the first and second recommended shows
        first_recommended_show_name = recommendations[0][0]
        second_recommended_show_name = recommendations[1][0]

        # Get the descriptions of the recommended shows
        first_recommended_show_description = next(
            show["description"]
            for show in tv_shows
            if show["name"] == first_recommended_show_name
        )
        second_recommended_show_description = next(
            show["description"]
            for show in tv_shows
            if show["name"] == second_recommended_show_name
        )

        # Create custom show suggestions
        custom_show1 = f"Show #1 is based on the fact that you loved the input shows. Its name is: {first_recommended_show_name} and it is about {first_recommended_show_description}."
        custom_show2 = f"Show #2 is based on the shows that I recommended for you. Its name is: {second_recommended_show_name} and it is about {second_recommended_show_description}."

        return custom_show1, custom_show2


# Cosine similarity function
def cosine_similarity(a, b):
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
    custom_show1, custom_show2 = generate_custom_shows(
        user_shows, recommendations, tv_shows
    )
    print(custom_show1)
    print(custom_show2)
    print("Here are also the 2 TV show ads. Hope you like them!")


if __name__ == "__main__":
    main()