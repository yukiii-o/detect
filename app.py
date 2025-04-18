import streamlit as st

# Streamlit page setup
st.set_page_config(page_title="AI Text Detector", page_icon="🤖", layout="centered")

st.title("AIDENTIFY - AI Text Detector")
st.markdown("Welcome! Paste a sentence or paragraph below and find out if it was **AI-generated** or **Human-written**.")

# Large input area
user_input = st.text_area("Input your text here:", height=200, placeholder="Type or paste your text...")

# Detection logic triggers on button click
if st.button("SCAN") and user_input.strip():
    # ====== Begin original code block ======

    import math
    # Dataset (need to modify to add more stuff here): 1 = human-written, 0 = AI-generated
    data = [
        #Human-written:
    ("The giant took a gulp of tea and wiped his mouth with the back of his hand.", 1),
    ("Taking him to buy his things tomorrow.", 1),
    ("She stopped to draw a deep breath and then went ranting on.", 1),
    ("It seemed she had been wanting to say all this for years. ", 1),
    ("Dunno if he had enough human left in him to die", 1),
    ("Although he could tell it was daylight, he kept his eyes shut tight. ", 1),
    ("There was suddenly a loud tapping noise. ", 1),
    ("He went straight to the window and jerked it open.", 1),
    ("And the people have no imagination", 1),
    ("He was standing before a garden, all a-bloom with roses.", 1),
    ("The little prince gazed at them.", 1),
    ("They all looked like his flower.", 1),
    ("And he was overcome with sadness.", 1),
    ("And he lay down in the grass and cried.", 1),
    ("It was then that the fox appeared.", 1),
    ("The fox gazed at the little prince, for a long time.", 1),
    ("The little prince went away, to look again at the roses.", 1),
    ("And the roses were very much embarrassed.", 1),
    ("And a second brilliantly lighted express thundered by, in the opposite direction.", 1),
    ("This was a merchant who sold pills that had been invented to quench thirst.", 1),


#AI written:
    ("This is a human-written sentence.", 0),
    ("I enjoy reading books and writing stories.", 0),
    ("Natural language processing is a fascinating field.", 0),
    ("I love spending time with my family and friends.", 0),
    ("She enjoys reading books on rainy days.", 0),
    ("He loves playing chess in the evening.", 0),
    ("They traveled to Paris for their honeymoon.", 0),
    ("I always drink coffee in the morning.", 0),
    ("She dances to her favorite songs at home.", 0),
    ("We watched a movie last night.", 0),
    ("The children are playing in the park.", 0),
    ("My dog loves to chase after the ball.", 0),
    ("The ocean waves crashed against the shore.", 0),
    ("He enjoys painting landscapes in his free time.", 0),
    ("We visited the museum yesterday.", 0),
    ("She always wears her favorite blue dress.", 0),
    ("The rain is falling softly outside.", 0),
    ("They had a wonderful dinner at a restaurant.", 0),
    ("I prefer reading books to watching movies.", 0),
    ("The flowers in the garden are blooming beautifully.", 0),
    ("He played the guitar at the concert.", 0),
    ("We went hiking in the mountains last weekend.", 0),
    ("The cake was delicious and freshly baked.", 0),
    ("She smiled as she saw the rainbow in the sky.", 0),
    ("They celebrated their anniversary at a fancy hotel.", 0),
    ("I finished my homework early today.", 0),
    ("He bought a new car last week.", 0),
    ("I love watching the sunset by the beach.", 0),
    ("She helped me with my project last night.", 0),
    ("The city lights looked beautiful from the rooftop.", 0),
    ("We went to a concert last Friday.", 0),
    ("She baked a pie for Thanksgiving.", 0),
    ("He read the newspaper while drinking his coffee.", 0),
    ("I like to take long walks in the evening.", 0),
    ("The cat sat on the windowsill, watching the birds.", 0),
    ("She went shopping for groceries this morning.", 0),
    ("They visited their grandparents over the weekend.", 0),
    ("We had a bonfire by the lake during our camping trip.", 0),
    ("She loves to explore new hiking trails.", 0),
    ("I spent the afternoon painting with watercolors.", 0),
    ("We watched a movie marathon on a rainy day.", 0),
    ("She took a yoga class to relieve stress.", 0),
    ("I spent the weekend organizing my closet.", 0),
    ("We had a barbecue in the backyard with friends.", 0),
    ("She loves to collect postcards from her travels.", 0),
    ("I spent the evening listening to music and relaxing.", 0),
    ("We visited the botanical garden to see the flowers.", 0),
    ("He played his favorite song on the piano.", 0),
    ("She spent the day volunteering at the animal shelter.", 0),
    ("We went to a new restaurant for dinner.", 0),
    ("I took a walk in the park to clear my mind.", 0),
    ("The candles created a warm, cozy atmosphere in the room.", 0),
    ("She enjoys watching documentaries on history.", 0),
    ("We had a picnic near the riverbank.", 0),
    ("I spent some time journaling about my day.", 0),
    ("She loves to ride her bike through the neighborhood.", 0),
    ("We went to the carnival and rode the Ferris wheel.", 0),
    ("He tried a new recipe for dinner and it turned out great.", 0),
    ("She spent the afternoon gardening in her backyard.", 0),
    ("We saw a beautiful rainbow after the storm.", 0),
    ("I took a photo of the flowers blooming in spring.", 0),
    ("She wore her favorite boots on the hike.", 0),
    ("We visited a local vineyard for a wine tasting.", 0),
    ("I love to watch the snowflakes fall during winter.", 0),
    ("He spent the afternoon reading a book at the café.", 0),
    ("She went to the market to buy fresh produce.", 0),
    ("We played a game of charades with our friends.", 0),
    ("He baked a cake to celebrate his birthday.", 0),
    ("We went for a drive along the coast.", 0),
    ("She enjoyed a relaxing day at the spa.", 0),
    ("I watched a beautiful documentary about wildlife.", 0),
    ("We had a game night with board games and snacks.", 0),
    ("She made a scrapbook of her travels around the world.", 0),
    ("I went to a concert and enjoyed the live music.", 0),
    ("They enjoyed a romantic dinner under the stars.", 0),
    ("He spent the day fishing at the lake.", 0),
    ("We went on a road trip and visited several cities.", 0),
    ("I sat by the fire and enjoyed a cup of tea.", 0),
    ("She took a relaxing bubble bath after a long day.", 0),
    ("We saw a movie and had popcorn for snacks.", 0),
    ("She loved making homemade candles for gifts.", 0),
    ("We took a walk in the forest and enjoyed the quiet.", 0),
    ("He played a new video game with his friends.", 0),
    ("We went to the beach to watch the sunrise.", 0),
    ("She spent the evening reading by candlelight.", 0),
     ]

    texts, labels = zip(*data)

    def preprocess(text):
        text = text.lower()
        text = "".join(char for char in text if char.isalpha() or char.isspace())
        return text

    def tokenize(text):
        return text.split()

    vocab = set()
    for text in texts:
        tokens = tokenize(preprocess(text))
        vocab.update(tokens)

    vocab = list(vocab)

    def text_to_vector(text):
        vector = [0] * len(vocab)
        tokens = tokenize(preprocess(text))
        for token in tokens:
            if token in vocab:
                vector[vocab.index(token)] += 1
        return vector

    X = [text_to_vector(text) for text in texts]
    y = list(labels)

    def train_test_split(X, y, test_size=0.25):
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    class NaiveBayesClassifier:
        def __init__(self):
            self.class_probs = {}
            self.feature_probs = {}

        def fit(self, X, y):
            n_samples = len(X)
            n_features = len(X[0])
            classes = set(y)

            for c in classes:
                self.class_probs[c] = sum(1 for label in y if label == c) / n_samples
                self.feature_probs[c] = [0] * n_features
                class_samples = [X[i] for i in range(n_samples) if y[i] == c]
                for feature_index in range(n_features):
                    feature_count = sum(sample[feature_index] for sample in class_samples)
                    self.feature_probs[c][feature_index] = (feature_count + 1) / (len(class_samples) + n_features)

        def predict(self, X):
            predictions = []
            for sample in X:
                max_prob = -math.inf
                best_class = -1
                for c in self.class_probs:
                    prob = math.log(self.class_probs[c])
                    for feature_index in range(len(sample)):
                        if sample[feature_index] > 0:
                            prob += math.log(self.feature_probs[c][feature_index])
                    if prob > max_prob:
                        max_prob = prob
                        best_class = c
                predictions.append(best_class)
            return predictions

        def predict_proba(self, X):
            probabilities = []
            for sample in X:
                class_probs = {}
                total_prob = 0
                for c in self.class_probs:
                    prob = self.class_probs[c]
                    for feature_index in range(len(sample)):
                        if sample[feature_index] > 0:
                            prob *= self.feature_probs[c][feature_index]
                    class_probs[c] = prob
                    total_prob += prob

                for c in class_probs:
                    class_probs[c] = class_probs[c] / total_prob
                probabilities.append(class_probs)
            return probabilities

    classifier = NaiveBayesClassifier()
    classifier.fit(X_train, y_train)

    def detect_ai_text(text):
        vector = text_to_vector(text)
        prediction = classifier.predict([vector])[0]
        proba = classifier.predict_proba([vector])[0]
        confidence = proba[prediction] * 100
        return ("AI-generated" if prediction == 1 else "Human-written", confidence)

    result, confidence = detect_ai_text(user_input)

    # Output section
    st.markdown("### 📊 Result:")
    st.success(f"**Prediction:** {result}")
    st.info(f"**Confidence:** {confidence:.2f}%")

else:
    st.markdown("Enter a sentence and click the button to get a prediction.")
