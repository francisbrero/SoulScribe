import random

# Get a random question from the seed_questions.txt file
def get_random_question():
    with open("seed_questions.txt", "r") as f:
        questions = f.readlines()
    return random.choice(questions).strip()