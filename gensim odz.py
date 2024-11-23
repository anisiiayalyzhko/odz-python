from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Вітання
print("Hello! I will help you determine the genre of a movie based on a short description of its plot.")

# Expanded movie descriptions and corresponding genres (more variety)
movie_descriptions = [
    "A young wizard joins a magical school and fights a dark lord. A princess discovers her magical powers in a kingdom filled with mythical creatures.",  # Fantasy
    "A group of friends go on a hilarious road trip. A comedian navigates the ups and downs of his career, leading to laugh-out-loud moments.",                # Comedy
    "A detective investigates a series of gruesome murders. A spy is on a mission to uncover a secret plot by a criminal mastermind.",         # Thriller
    "Aliens visit Earth and a scientist tries to communicate with them. A group of astronauts embarks on a dangerous mission to save the planet. In the future, a robot and a scientist discover the secret to saving humanity. An intergalactic war breaks out between warring alien factions.",  # Sci-Fi
    "A love story set during a historical war. A couple finds love in the middle of a fierce civil war. A tragic love story unfolds against the backdrop of a world war.",                      # Romance
    "A haunted house traps a family with terrifying ghosts. A monster terrorizes a small town in a remote village. A group of friends find themselves trapped in a haunted mansion.",          # Horror
    "A superhero saves the world from a powerful villain. A heroic warrior defeats a tyrannical ruler to restore peace.",            # Action
    "A team of adventurers sets out on a quest to find a legendary treasure. An archaeologist embarks on a perilous journey to discover ancient artifacts.", # Adventure
    "A small town cop takes on a powerful drug cartel. A detective uncovers a citywide conspiracy while solving a murder.",               # Crime
]

genres = [
    "Fantasy", "Comedy", "Thriller", "Sci-Fi", "Romance", "Horror", "Action", "Adventure",
    "Crime"
]

# Попередня обробка тексту
stop_words = set(stopwords.words('english'))
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens

processed_descriptions = [preprocess(desc) for desc in movie_descriptions]

# Створення словника та корпусу
dictionary = corpora.Dictionary(processed_descriptions)
corpus = [dictionary.doc2bow(text) for text in processed_descriptions]

# Побудова LDA-моделі
num_topics = len(genres)  # Відповідає кількості жанрів
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# Виведення тематик
print("\nThe model is trained! Here are some key words for each topic:")
for idx, topic in lda_model.show_topics(formatted=False):
    print(f"Topic {idx + 1}: {[word for word, _ in topic]}")

# Запит нового опису
print("\nDescribe the plot of a movie very shortly, and I will try to determine its genre:")
new_description = input().strip()
new_bow = dictionary.doc2bow(preprocess(new_description))
topic_distribution = lda_model.get_document_topics(new_bow)

# Визначення ймовірностей
print("\nGenre probabilities:")
for topic_id, probability in topic_distribution:
    genre = genres[topic_id]
    print(f"Genre '{genre}': {probability * 100:.2f}%")

# Визначення найбільш ймовірного жанру
most_probable_topic = max(topic_distribution, key=lambda x: x[1])
predicted_genre = genres[most_probable_topic[0]]
print(f"\nIn my opinion, this movie most likely belongs to the genre: {predicted_genre}")

# Прощання
print("\nThank you for using this tool! Have a great day!")
