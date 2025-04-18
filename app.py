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
    ("The AI-generated content is becoming more advanced.", 1),
    ("Machine learning models can generate realistic text.", 1),
    ("AI-generated text is often indistinguishable from human writing.", 1),
    ("In the course of this life I have had a great many encounters with a great many people who have been concerned with matters of consequence.", 1),
("If one gets lost in the night, such knowledge is valuable.", 1),
("Then I would never talk to that person about boa constrictors, or primeval forests, or stars.", 1),
("Remember, I had crashed in the desert a thousand miles from any inhabited region.", 1),
("When a mystery is too overpowering, one dare not disobey.", 1),
("But I had never drawn a sheep.", 1),
("So I drew for him one of the two pictures I had drawn so often. I", 1),
("My friend smiled gently and indulgently.", 1),
("So then I did my drawing over once more.", 1),
("But it was rejected too, just like the others.", 1),
("And I threw out an explanation with it.", 1),
("He bent his head over the drawing", 1),
("And he sank into a reverie, which lasted a long time.", 1),
("You can imagine how my curiosity was aroused by this half-confidence about the others", 1),
("But he was in Turkish costume, and so nobody would believe what he said.", 1),
("One must not hold it against them.", 1),
("To those who understand life, that would have given a much greater air of truth to my story.", 1),
("For I do not want any one to read my book carelessly", 1),
("Six years have already passed since my friend went away from me, with his sheep.", 1),
("It is for that purpose, again, that I have bought a box of paints and some pencils.", 1),
("But I am not at all sure of success.", 1),
("And I feel some doubts about the color of his costume.", 1),
("He thought, perhaps, that I was like himself.", 1),
("Perhaps I am a little like the grown-ups.", 1),
("He thought, perhaps, that I was like himself. ", 1),
("My friend never explained anything to me.", 1),
("This time, once more, I had the sheep to thank for it.", 1),
("The idea of the herd of elephants made the little prince laugh.", 1),
("That would be very useful to them if they were to travel some day.", 1),
("But with the others I have not been successful", 1),
(" I was carried beyond myself by the inspiring force of urgent necessity.", 1),
("At first you seemed to be very much surprised.", 1),
(". And then you laughed to yourself.", 1),
("Everybody knows that when it is noon in the United States the sun is setting over France.", 1),
("But the little prince made no reply", 1),
("At that moment I was very busy trying to unscrew a bolt that had got stuck in my engine", 1),
("And I had so little drinking-water left that I had to fear for the worst.", 1),
("The little prince never let go of a question, once he had asked it.", 1),
("There was a moment of complete silence.", 1),
("Flowers are weak creatures.", 1),
("That made me a little ashamed.", 1),
("He tossed his golden curls in the breeze.", 1),
("He has never smelled a flower", 1),
("He has never done anything in his life but add up figures.", 1),
("The little prince was now white with rage.", 1),
("He could not say anything more. ", 1),
("His words were choked by sobbing.", 1),
("I had let my tools drop from my hands", 1),
("I took him in my arms, and rocked him.", 1),
("I did not know what to say to him.", 1),
("I did not know how I could reach him.", 1),
("I soon learned to know this flower better.", 1),
("She chose her colors with the greatest care.", 1),
("She did not wish to go out into the world all rumpled, like the field poppies.", 1),
("But she interrupted herself at that point.", 1),
("She had come in the form of a seed.", 1),
("She cast her fragrance and her radiance over me.", 1),
("He carefully cleaned out his active volcanoes.", 1),
("On our earth we are obviously much too small to clean out our volcanoes.", 1),
("He was surprised by this absence of reproaches.", 1),
("For she did not want him to see her crying.", 1),
("The first of them was inhabited by a king.", 1),
("He did not know how the world is simplified for kings.", 1),
("It is years since I have seen anyone yawning.", 1),
("He sputtered a little, and seemed vexed.", 1),
("For what the king fundamentally insisted upon was that his authority should be respected.", 1),
("The king hastened to assure him.", 1),
("The second planet was inhabited by a conceited man.", 1),
("The little prince clapped his hands.", 1),
("The conceited man raised his hat in a modest salute.", 1),
("The conceited man again raised his hat in salute.", 1),
("After five minutes of this exercise the little prince grew tired of the game's monotony.", 1),
("But the conceited man did not hear him", 1),
("Conceited people never hear anything but praise.", 1),
("The next planet was inhabited by a tippler.", 1),
("And the little prince went away, puzzled.", 1),
("The fourth planet belonged to a businessman.", 1),
("The businessman raised his head.", 1),
("Nevertheless, he still had some more questions.", 1),
("The little prince was still not satisfied.", 1),
("The businessman opened his mouth, but he found nothing to say in answer.", 1),
("There was just enough room on it for a street lamp and a lamplighter.", 1),
("For it is possible for a man to be faithful and lazy at the same time.", 1),
("The sixth planet was ten times larger than the last one", 1),
("He does not leave his desk.", 1),
("Because an explorer who told lies would bring disaster on the books of the geographer.", 1),
("Because intoxicated men see double.", 1),
("That would be too complicated.", 1),
("The geographer was suddenly stirred to excitement.", 1),
("And, having opened his big register, the geographer sharpened his pencil.", 1),
("Two volcanoes are active and the other is extinct.", 1),
("And the little prince went away, thinking of his flower.", 1),
("So then the seventh planet was the Earth.", 1),
("Seen from a slight distance, that would make a splendid spectacle.", 1),
(". Having set their lamps alight, these would go off to sleep.", 1),
("It would be magnificent.", 1),
("They adore figures, and that will please them.", 1),
("They fancy themselves as important as the baobabs.", 1),
("The grown-ups, to be sure, will not believe you when you tell them that.", 1),
("None of them noticed a large tawny owl flutter past the window.", 1),
("Mr Dursley blinked and stared at the cat.", 1),
("In fact, it was nearly midnight before the cat moved at all. ", 1),
("This man's name was Albus Dumbledore.", 1),
("The nearest street lamp went out with a little pop.", 1),
("He couldn't kill that little boy.", 1),
(" You couldn't find two people who are less like us.", 1),
(" He was almost twice as tall as a normal man and at least five times as wide.", 1),
("In his vast, muscular arms he was holding a bundle of blankets. ", 1),
(" Inside, just visible, was a baby boy, fast asleep.", 1),
("Dumbledore turned and walked back down the street.", 1),
(" He could just see the bundle of blankets on the step of number four", 1),
("It was a very sunny Saturday and the zoo was crowded with families.", 1),
("Dudley quickly found the largest snake in the place.", 1),
("He shuffled away.", 1),
("The snake jabbed its tail at a little sign next to the glass.", 1),
("There were no photographs of them in the house. ", 1),
("He couldn't remember his parents at all.", 1),
("They also carried knobbly sticks, used for hitting each other while the teachers weren't looking.", 1),
("This was supposed to be good training for later life. ", 1),
("He chuckled at his own joke. ", 1),
("He sat down on the bed and staredaround him. ", 1),
("Other shelves were full of books.", 1),
("They were the only things in the room that looked as though they'd never been touched. ", 1),
("Next morning at breakfast, everyone was rather quiet.", 1),
("Uncle Vernon stayed at home again.", 1),
("They didn't stop to eat or drink all day.", 1),
("He'd never had such a bad day in his life.", 1),
("Exactly what he was looking for, none of them knew.", 1),
("I want to stay somewhere with a television.", 1),
("It was very cold outside the car.", 1),
("It was freezing in the boat.", 1),
("There were only two rooms.", 1),
("He was in a very good mood.", 1),
("As night fell, the promised storm blew up around them.", 1),
("Someone was outside, knocking to come in. ", 1),
("He was holding a rifle in his hands.", 1),
("A giant of a man was standing in the doorway.", 1),
("Uncle Vernon made a funny rasping noise. ", 1),
("He held out an enormous hand and shook Harry's whole arm. ", 1),
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
("We had a picnic in the park yesterday.", 0),
("The sky turned orange as the sun set.", 0),
("She loves gardening and grows her own vegetables.", 0),
("I found a beautiful seashell on the beach.", 0),
("We met some old friends at the coffee shop.", 0),
("She finished reading her book in one sitting.", 0),
("The bird sang a cheerful song from the tree.", 0),
("They decided to take a road trip across the country.", 0),
("I like to take photos of interesting clouds.", 0),
("She wore a red scarf to keep warm.", 0),
("He spent the afternoon working on his car.", 0),
("The stars were shining brightly in the night sky.", 0),
("We enjoyed a quiet evening by the fireplace.", 0),
("The ice cream truck passed by the street.", 0),
("She loves to write poetry in her journal.", 0),
("He's always telling funny jokes to make us laugh.", 0),
("I walked through the forest and admired the trees.", 0),
("They celebrated Christmas with their family.", 0),
("The wind whispered through the tall grass.", 0),
("She made a delicious pasta for dinner.", 0),
("I bought a new pair of shoes today.", 0),
("He enjoys playing soccer with his friends.", 0),
("We spent the day exploring a new city.", 0),
("She sang her favorite song at karaoke.", 0),
("The candles flickered in the dark room.", 0),
("He painted a portrait of his mother.", 0),
("I took a nap in the afternoon after a long day.", 0),
("The mountains were covered in snow.", 0),
("She decorated her house with beautiful flowers.", 0),
("We saw a shooting star while camping.", 0),
("He enjoyed a cup of tea in the morning.", 0),
("The sound of the rain was very soothing.", 0),
("She took a stroll along the beach in the evening.", 0),
("We ate dinner at a cozy little bistro.", 0),
("The smell of fresh bread filled the air.", 0),
("She painted a beautiful sunset on canvas.", 0),
("He likes to read mystery novels before bed.", 0),
("We took a boat ride across the lake.", 0),
("The treehouse was our favorite childhood spot.", 0),
("See watched the clouds drift by from the porch.", 0),
("I enjoyed a hot cup of cocoa by the fire.", 0),
("They had a barbecue in the backyard.", 0),
("The train ride was scenic and relaxing.", 0),
("He made a toast to celebrate their success.", 0),
("We built a sandcastle on the beach.", 0),
("She loves listening to classical music in the evening.", 0),
("The puppy chased after its tail in circles.", 0),
("I found a quiet spot to read my book.", 0),
("She went for a jog in the park this morning.", 0),
("We watched a beautiful sunrise from the balcony.", 0),
("The coffee smelled so good in the morning.", 0),
("She enjoys knitting scarves during the winter.", 0),
("He worked late into the night on his project.", 0),
("We spent the afternoon at the zoo.", 0),
("She went to a pottery class last weekend.", 0),
("He played video games with his friends online.", 0),
("The house was decorated with colorful lights.", 0),
("I spent the morning cleaning the house.", 0),
("She took a break to relax in the hammock.", 0),
("They went to the mountains for a weekend getaway.", 0),
("I met a stranger who was kind and helpful.", 0),
("The breeze was cool and refreshing in the evening.", 0),
("She wore a yellow dress to the party.", 0),
("We enjoyed some fresh fruit for breakfast.", 0),
("The sound of the ocean was calming and peaceful.", 0),
("He visited the art gallery for inspiration.", 0),
("She made a scrapbook of her favorite memories.", 0),
("We went ice skating on the frozen lake.", 0),
("The flowers in the garden smelled sweet and fragrant.", 0),
("He enjoys writing stories in his free time.", 0),
("They enjoyed a relaxing afternoon at the spa.", 0),
("I took a photograph of the sunset on my phone.", 0),
("She baked a loaf of banana bread this morning.", 0),
("We saw a deer while walking through the forest.", 0),
("The weather was perfect for a picnic in the park.", 0),
("I love listening to rain sounds while I sleep.", 0),
("She loved the smell of fresh flowers in her room.", 0),
("We visited a historical site during our trip.", 0),
("He surprised her with a beautiful bouquet of flowers.", 0),
("The sound of the guitar echoed through the room.", 0),
("We played board games all afternoon.", 0),
("She enjoyed a peaceful moment by the lake.", 0),
("I tried a new recipe for dinner tonight.", 0),
("He bought a new guitar and started learning to play.", 0),
("We went camping and made s'mores around the fire.", 0),
("The birds were chirping early in the morning.", 0),
("She painted a mural on the wall of her room.", 0),
("I love walking through the woods in autumn.", 0),
("We explored an old castle during our vacation.", 0),
("The puppy fell asleep in my lap.", 0),
("She enjoys visiting new cafes around the city.", 0),
("We went bowling with some friends last weekend.", 0),
("He spent the day at the beach, reading a book.", 0),
("I made a cup of herbal tea to relax.", 0),
("She took a picture of the sunset with her camera.", 0),
("We enjoyed a delicious lunch at a local diner.", 0),
("He loves collecting old coins from different countries.", 0),
("The stars were so bright on the clear night.", 0),
("I spent the evening working on a puzzle.", 0),
("She wore a cozy sweater to keep warm.", 0),
("They went to a festival to enjoy the food and music.", 0),
("We went for a bike ride through the countryside.", 0),
("I like to sit by the window and watch the rain.", 0),
("She loves baking cakes for her friends and family.", 0),
("We danced to our favorite songs at the party.", 0),
("He took a photo of the full moon last night.", 0),
("I visited the bookstore to find a new novel.", 0),
("She made a homemade pizza for dinner.", 0),
("They went to the lake to enjoy some time outdoors.", 0),
("We saw a beautiful butterfly in the garden.", 0),
("She went for a run in the morning to stay fit.", 0),
("I enjoy having a cup of hot chocolate on cold days.", 0),
("He practiced his piano skills every evening.", 0),
("The autumn leaves turned golden and fell to the ground.", 0),
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
