

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# ===============================
# 1. Synthetic Dataset (for demo)
# ===============================
texts = [
    # Depression
    "I feel so empty and hopeless.", "I have been crying for no reason.",
    "I can't concentrate on my work anymore.", "Life doesn’t seem worth living.",
    "Sometimes I feel like a burden to others.", "Nothing excites me anymore.",
    "I struggle to get out of bed every morning.", "I don’t enjoy the things I used to love.",
    "I feel worthless and unloved.", "I feel my future is hopeless.",
    # Anxiety
    "My chest feels tight and I can't breathe.", "I'm scared something bad will happen.",
    "I worry about everything all the time.", "My hands shake when I get nervous.",
    "I overthink every small detail.", "I always expect the worst to happen.",
    "I get panic attacks in crowded places.", "I feel restless all the time.",
    "I can’t sleep because I keep worrying.", "I feel something bad is about to happen.",
    # Stress
    "Work deadlines are stressing me out.", "I can't sleep because of constant tension.",
    "My mind is always racing with worries.", "I feel pressure from too many responsibilities.",
    "I feel overwhelmed with tasks.", "I get irritated easily these days.",
    "I’m always rushing to meet deadlines.", "My work-life balance is terrible.",
    "I feel burned out from everything.", "I can’t focus because of constant stress.",
    # Normal
    "I enjoy spending time with my friends.", "Today was a great day, I feel relaxed.",
    "I'm excited about my new job opportunity.", "I love going out and meeting people.",
    "I feel happy and grateful today.", "I enjoyed my walk in the park.",
    "I had a nice conversation with my family.", "I feel peaceful and content.",
    "I feel motivated to work today.", "I’m happy with how life is going."
]

labels = (
    ["depression"] * 10 +
    ["anxiety"] * 10 +
    ["stress"] * 10 +
    ["normal"] * 10
)

df = pd.DataFrame({"text": texts, "label": labels})

# ===============================
# 2. Preprocessing (TF-IDF + SMOTE)
# ===============================
X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Handle imbalanced dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_vec, y)

# ===============================
# 3. Model Training
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# 4. Prediction Function
# ===============================
def predict_text(text: str) -> str:
    """
    Predicts the mental health category of input text.
    Categories: depression, anxiety, stress, normal
    """
    text_features = vectorizer.transform([text])
    prediction = model.predict(text_features)[0]
    return prediction

# ===============================
# 5. Interactive CLI Mode
# ===============================
if __name__ == "__main__":
    print("\n🧠 Mental Health Chat Analysis Ready! (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Exiting. Take care of your mental health!")
            break
        result = predict_text(user_input)
        print(f"🤖 Prediction: {result}")
