
# ============================================================
# UCS420 - COGNITIVE COMPUTING
# ASSIGNMENT 10
# DIABETES CARE INFORMATION ASSISTANT
#
# Name: Madhav Narang
# Roll No: 1024170287
# ============================================================

# STEP 1: INSTALL AND IMPORT LIBRARIES

!pip -q install nltk scikit-learn pandas matplotlib seaborn ipywidgets

import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import ipywidgets as widgets
from IPython.display import display, clear_output

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("Libraries imported successfully!")


# STEP 2: STUDENT INFORMATION

STUDENT_NAME = "Madhav Narang"
ROLL_NUMBER = "1024170287"

print("\n==========================================")
print("DIABETES CARE INFORMATION ASSISTANT")
print("==========================================")
print("Name:", STUDENT_NAME)
print("Roll No:", ROLL_NUMBER)
print("Subject: Cognitive Computing UCS420")


# STEP 3: LABELLED DATASET
# 48 queries across 8 intent categories

data = {

    "glucose_monitoring": [
        "How can I monitor my blood sugar levels?",
        "When should I check my glucose level?",
        "What is a blood glucose meter?",
        "How do I record my sugar readings?",
        "Why should I track my blood sugar daily?",
        "How can I understand my glucose readings?"
    ],

    "a1c": [
        "What is HbA1c in diabetes?",
        "How often should I check my A1c?",
        "What does an A1c test measure?",
        "Why is the A1c test important?",
        "How is average blood sugar measured?",
        "What is the purpose of glycated hemoglobin testing?"
    ],

    "nutrition": [
        "What foods are suitable for diabetes?",
        "Can people with diabetes eat fruits?",
        "Which foods contain carbohydrates?",
        "How can I plan a healthy diabetic diet?",
        "Why should I avoid sugary drinks?",
        "What are healthy breakfast options for diabetes?"
    ],

    "exercise": [
        "What exercises are suitable for diabetes?",
        "Can walking help manage blood sugar?",
        "How often should a diabetic person exercise?",
        "What are the benefits of physical activity?",
        "Can I do light exercise after meals?",
        "How does exercise affect glucose levels?"
    ],

    "medication": [
        "Why are diabetes medicines prescribed?",
        "What is the purpose of metformin?",
        "How should I manage my diabetes medicines?",
        "What is insulin used for?",
        "Can I stop taking my diabetes medicine?",
        "Why is medication adherence important?"
    ],

    "symptoms": [
        "What are common symptoms of diabetes?",
        "Why do diabetic patients feel thirsty?",
        "What causes frequent urination in diabetes?",
        "Can diabetes cause tiredness?",
        "What are the early warning signs of diabetes?",
        "Why can diabetes cause increased hunger?"
    ],

    "complications": [
        "What complications can diabetes cause?",
        "How does diabetes affect the kidneys?",
        "Can diabetes affect eyesight?",
        "Why is foot care important in diabetes?",
        "Can diabetes affect the nerves?",
        "How can diabetes affect heart health?"
    ],

    "foot_care": [
        "How should diabetic patients care for their feet?",
        "Why should I inspect my feet daily?",
        "What should I do about a foot wound?",
        "Why is proper footwear important for diabetes?",
        "How can I prevent diabetic foot problems?",
        "When should a diabetic patient visit a foot specialist?"
    ]
}


# Convert dictionary into labelled corpus

texts = []
labels = []

for intent, queries in data.items():
    for query in queries:
        texts.append(query)
        labels.append(intent)

df = pd.DataFrame({
    "Query": texts,
    "Intent": labels
})

print("\nDATASET")
print(df.head(10))

print("\nTotal Queries:", len(df))
print("Total Intent Categories:", df["Intent"].nunique())

print("\nQueries per Category:")
print(df["Intent"].value_counts())


# STEP 4: HIERARCHICAL TAXONOMY

taxonomy = {

    "Monitoring": {
        "Glucose Monitoring": "glucose_monitoring",
        "HbA1c Testing": "a1c"
    },

    "Lifestyle Management": {
        "Healthy Nutrition": "nutrition",
        "Physical Activity": "exercise"
    },

    "Treatment": {
        "Medication Management": "medication",
        "Foot Care": "foot_care"
    },

    "Health Concerns": {
        "Symptoms": "symptoms",
        "Complications": "complications"
    }
}

print("\nHIERARCHICAL TAXONOMY")

for major, subcategories in taxonomy.items():

    print("\n", major)

    for subcategory in subcategories:
        print("   ->", subcategory)


# STEP 5: DOMAIN CONCEPTS AND RELATIONSHIPS

knowledge_graph = {

    "Diabetes": [
        ("Diabetes", "requires_monitoring", "Blood Glucose"),
        ("Diabetes", "associated_with", "Insulin"),
        ("Diabetes", "monitored_by", "HbA1c")
    ],

    "Blood Glucose": [
        ("Blood Glucose", "measured_by", "Glucose Meter"),
        ("Blood Glucose", "influenced_by", "Food"),
        ("Blood Glucose", "influenced_by", "Exercise")
    ],

    "Insulin": [
        ("Insulin", "regulates", "Blood Glucose"),
        ("Insulin", "used_in", "Diabetes Treatment")
    ],

    "HbA1c": [
        ("HbA1c", "indicates", "Average Blood Glucose")
    ],

    "Food": [
        ("Food", "contains", "Carbohydrates"),
        ("Carbohydrates", "affect", "Blood Glucose")
    ],

    "Exercise": [
        ("Exercise", "supports", "Glucose Management")
    ],

    "Diabetes Complications": [
        ("Diabetes", "may_lead_to", "Kidney Problems"),
        ("Diabetes", "may_lead_to", "Eye Problems"),
        ("Diabetes", "may_lead_to", "Nerve Problems")
    ]
}

print("\nDOMAIN KNOWLEDGE GRAPH")

all_relations = []

for concept, relations in knowledge_graph.items():

    for relation in relations:
        all_relations.append(relation)
        print(relation)

print("\nTotal Relationships:", len(all_relations))


# STEP 6: NLP PREPROCESSING

stop_words = set(stopwords.words("english"))

# Preserve negation words because they can change meaning.
stop_words -= {"no", "not", "nor", "never"}

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    # Lowercase conversion
    text = text.lower()

    # Tokenization
    tokens = re.findall(r"[a-z]+", text)

    # Stopword removal and lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


df["Processed_Query"] = df["Query"].apply(preprocess_text)

print("\nNLP PREPROCESSING EXAMPLES")

print(
    df[["Query", "Processed_Query"]].head(10)
)


# STEP 7: TF-IDF FEATURE EXTRACTION

X = df["Processed_Query"]
y = df["Intent"]

vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(X)

print("\nTF-IDF FEATURE MATRIX")
print("Shape:", X_tfidf.shape)


# STEP 8: TRAIN AND TEST DATA

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# STEP 9: TRAIN NAIVE BAYES CLASSIFIER

model = MultinomialNB()

model.fit(X_train, y_train)

print("\nNaive Bayes Model Trained Successfully!")


# STEP 10: MODEL EVALUATION

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)

recall = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)

f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

print("\nMODEL PERFORMANCE")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1, 3))

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

plt.figure(figsize=(11, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Diabetes Assistant - Confusion Matrix")
plt.xlabel("Predicted Intent")
plt.ylabel("Actual Intent")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# STEP 11: DOMAIN-SPECIFIC KNOWLEDGE BASE

knowledge_base = {

    "glucose_monitoring": """
    Blood glucose monitoring helps people understand
    their glucose patterns.

    A glucose meter measures blood sugar using a
    small blood sample.

    Follow the monitoring schedule recommended
    by your healthcare professional.
    """,

    "a1c": """
    HbA1c is a blood test that reflects average
    blood glucose over approximately the previous
    two to three months.

    Your healthcare professional can recommend
    an appropriate testing schedule and target.
    """,

    "nutrition": """
    A diabetes-friendly eating pattern can include
    vegetables, whole grains, legumes, suitable
    fruits and balanced portions.

    Limit sugary drinks and discuss an individual
    meal plan with a registered dietitian.
    """,

    "exercise": """
    Regular physical activity can support blood
    glucose management and overall health.

    Walking, cycling and other suitable activities
    may be helpful.

    Choose activity according to your health and
    your healthcare professional's advice.
    """,

    "medication": """
    Diabetes medicines help manage blood glucose.

    Metformin is commonly prescribed for type 2
    diabetes, while insulin is essential for people
    who require it.

    Never stop, start or change medication doses
    without consulting your healthcare professional.
    """,

    "symptoms": """
    Common symptoms associated with diabetes
    include increased thirst, frequent urination,
    increased hunger, tiredness and blurred vision.

    Symptoms alone cannot confirm diabetes.
    Consult a healthcare professional for testing.
    """,

    "complications": """
    Diabetes can increase the risk of complications
    affecting the eyes, kidneys, nerves, heart and
    blood vessels.

    Regular medical checkups and recommended
    preventive care are important.
    """,

    "foot_care": """
    Check your feet regularly for cuts, blisters,
    swelling, redness or wounds.

    Wear comfortable footwear and avoid walking
    barefoot.

    Seek prompt medical advice for any foot wound,
    especially if it is not healing.
    """
}


# STEP 12: INTENT DESCRIPTIONS

intent_descriptions = {

    "glucose_monitoring": "Blood glucose monitoring information",

    "a1c": "HbA1c testing and average glucose information",

    "nutrition": "Food, diet and nutrition information",

    "exercise": "Physical activity and exercise information",

    "medication": "Diabetes medication information",

    "symptoms": "Common diabetes symptoms information",

    "complications": "Diabetes-related complications information",

    "foot_care": "Diabetic foot care information"
}


# STEP 13: RULE-BASED REASONING

emergency_keywords = [

    "unconscious",
    "not waking",
    "seizure",
    "convulsion",
    "cannot breathe",
    "difficulty breathing",
    "chest pain",
    "fainting",
    "passed out",
    "severe confusion",
    "vomiting blood"
]

emergency_response = """
EMERGENCY ALERT!

Your message mentions a potentially serious symptom.

Please seek immediate medical help. In India, call
112 or your local emergency number.

Do not drive yourself.

If someone is unconscious or having a seizure,
do not give them food, drink or oral medicine.

This assistant cannot diagnose emergencies.
"""


domain_keywords = {

    "diabetes", "diabetic", "glucose", "sugar",
    "insulin", "hba1c", "a1c", "blood",
    "carbohydrate", "carbs", "diet", "food",
    "exercise", "walking", "medicine", "medication",
    "metformin", "tablet", "feet", "foot",
    "kidney", "eyesight", "thirst", "urination",
    "tiredness", "hunger", "wound", "blister",
    "meter", "monitoring", "hypoglycemia",
    "hyperglycemia", "fruit", "breakfast",
    "physical", "activity", "nerve", "heart",
    "complication", "meal", "healthy"
}


def is_emergency(query):

    query = query.lower()

    return any(
        keyword in query
        for keyword in emergency_keywords
    )


def is_supported_query(query):

    words = set(re.findall(r"[a-z]+", query.lower()))

    return bool(words.intersection(domain_keywords))


# STEP 14: COGNITIVE ASSISTANT

def diabetes_assistant(query):

    query = query.strip()

    if not query:
        return "Please enter a question."

    # Rule 1: Emergency escalation
    if is_emergency(query):

        return emergency_response

    # Rule 2: Unsupported query handling
    if not is_supported_query(query):

        return """
        Sorry, this query is outside my diabetes-care
        knowledge domain.

        Please ask about:
        - Blood glucose monitoring
        - HbA1c
        - Nutrition
        - Exercise
        - Medication
        - Symptoms
        - Complications
        - Foot care
        """

    # NLP preprocessing
    processed = preprocess_text(query)

    # Rule 3: Low-confidence handling
    if len(processed.split()) < 2:

        return """
        I could not understand the question properly.

        Please provide more details about your
        diabetes-care question.
        """

    # TF-IDF transformation
    query_vector = vectorizer.transform([processed])

    # Intent prediction
    probabilities = model.predict_proba(query_vector)[0]

    best_index = np.argmax(probabilities)

    confidence = probabilities[best_index]

    predicted_intent = model.classes_[best_index]

    # Low-confidence threshold
    if confidence < 0.20:

        return """
        I am not sufficiently confident about
        understanding this question.

        Please rephrase your question or consult
        a healthcare professional.
        """

    # Retrieve relevant information
    response = knowledge_base[predicted_intent]

    # Rule 4: Medication safety
    if predicted_intent == "medication":

        response += """

        MEDICATION SAFETY:
        Do not change or stop prescribed medication
        without consulting your healthcare provider.
        """

    # Rule 5: General medical safety
    response += """

    IMPORTANT:
    This is general educational information.
    It is not a medical diagnosis or a replacement
    for professional medical advice.
    """

    response += f"""

    Detected Intent: {predicted_intent}

    Intent Confidence: {confidence * 100:.2f}%
    """

    return response


# STEP 15: TESTING THE ASSISTANT

test_queries = [

    "How can I monitor my blood sugar?",
    "What is HbA1c?",
    "Which fruits are suitable for diabetes?",
    "Can walking help control glucose?",
    "What is metformin used for?",
    "Why do diabetic patients feel thirsty?",
    "Can diabetes damage the kidneys?",
    "How should I care for my diabetic feet?",
    "Who won the cricket match?",
    "Tell me about blue elephants",
    "Sugar",
    "I have chest pain and difficulty breathing"
]

print("\nASSISTANT TESTING")

for query in test_queries:

    print("\nUSER:", query)

    print("ASSISTANT:", diabetes_assistant(query))


# STEP 16: INTERACTIVE COLAB INTERFACE
# Supports multiple questions, one per line.

print("\n==========================================")
print("INTERACTIVE DIABETES CARE ASSISTANT")
print("==========================================")

print("Name:", STUDENT_NAME)
print("Roll No:", ROLL_NUMBER)


query_box = widgets.Textarea(
    value="",
    placeholder="Enter one or more questions, one per line...",
    description="Your Query:",
    layout=widgets.Layout(width="100%", height="120px")
)

ask_button = widgets.Button(
    description="Ask Assistant",
    button_style="success",
    icon="comments"
)

output_area = widgets.Output()


def ask_questions(button):

    with output_area:

        clear_output()

        queries = query_box.value.strip().splitlines()

        if not queries:

            print("Please enter at least one question.")
            return

        for query in queries:

            if query.strip():

                print("\nYOU:", query)

                print("\nASSISTANT:")
                print(diabetes_assistant(query))

                print("\n" + "-" * 60)


ask_button.on_click(ask_questions)

display(query_box)
display(ask_button)
display(output_area)


# ============================================================
# END OF ASSIGNMENT 10
# ============================================================

print("\nAssignment 10 Loaded Successfully!")
print("Student:", STUDENT_NAME)
print("Roll No:", ROLL_NUMBER)
