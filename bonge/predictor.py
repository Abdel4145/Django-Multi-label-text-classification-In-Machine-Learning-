from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from clean_data_helper import load_and_clean_data


df = load_and_clean_data()


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.TITLE).toarray()
labels = df.category_id



# Naive Bayes Classifier

x_train, x_test, y_train, y_test = train_test_split(df['TITLE'], df['CATEGORY'], random_state = 0, train_size=0.7, test_size=0.3)
count_vectorizer = CountVectorizer()
x_train_occurrences = count_vectorizer.fit_transform(x_train)
tfidf_transformer = TfidfTransformer()
x_tfidf = tfidf_transformer.fit_transform(x_train_occurrences)
model = LinearSVC().fit(x_tfidf, y_train)


while True:
    data = input("Please enter a news headline: \n")

    if data == "":
        print("Please enter an appropriate news headline: \n")
        continue
    else:
        break

prediction = model.predict(count_vectorizer.transform([data])[0])

if prediction[0] == 'b':
    print("Business")
elif prediction[0] == 't':
    print("Science")
elif prediction[0] == 'e':
    print("Entertainment")
elif prediction[0] == 'm':
    print("Health")






