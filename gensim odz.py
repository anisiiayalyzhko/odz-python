from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt_tab')

# Вітання
print("Hello! I will help you determine the genre of a movie based on a short description of its plot.")

# Жанри
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

# Завантаження описів фільмів з файлу
input_file = "movie_descriptions.txt"  # Назва файлу
try:
    with open(input_file, 'r', encoding='utf-8') as file:
        movie_descriptions = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found. Please ensure the file exists and try again.")
    exit()

# Перевірка, чи є в файлі описи
if len(movie_descriptions) < len(genres):
    print(f"Error: Insufficient movie descriptions in the file. At least {len(genres)} are required.")
    exit()

# Попередня обробка описів
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
if topic_distribution:
    most_probable_topic = max(topic_distribution, key=lambda x: x[1])
    predicted_genre = genres[most_probable_topic[0]]
    print(f"\nIn my opinion, this movie most likely belongs to the genre: {predicted_genre}")
else:
    print("\nSorry, I could not determine the genre of the movie.")

# Прощання
print("\nThank you for using this tool! Have a great day!")

