from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
import pandas as pd
import string
import os


def load_and_clean_data(): 
    # Loading and cleaning dataset
    df = pd.read_csv(os.path.join('F:/PYCHARM/DJANGO/Text Classification/ujasi/bonge/dataset', 'headlines.csv'))
    print(len(df))
    df.drop(df.tail(390000).index, inplace=True)
    df = df[['CATEGORY', 'TITLE']]
    df = df[pd.notnull(df['TITLE'])]
    print(df['TITLE'].head(5))
    df.TITLE = df.TITLE.apply(lambda x: x.lower())
    df.TITLE = df.TITLE.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df.TITLE = df.TITLE.apply(lambda x: x.translate(str.maketrans('', '', '1234567890')))
    df['category_id'] = df['CATEGORY'].factorize()[0]
    return df


def home(request):
    return render(request, 'bonge/home.html')


def predict(request):
    if request.method == 'POST':
        usertext = request.POST['usertext']
        print("Data received")
        print(usertext)
    else:
        print("No Data received")


    df = load_and_clean_data()
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin_1', ngram_range=(1,2), stop_words='english')
    features = tfidf.fit_transform(df.TITLE).toarray()
    labels = df.category_id

    # Train test split

    x_train, x_test, y_train, y_test = train_test_split(df['TITLE'], df['CATEGORY'], random_state=0, train_size=0.7, test_size=0.3)
    count_vectorizer = CountVectorizer()
    x_train_occurences = count_vectorizer.fit_transform(x_train)
    tfidf_transformer = TfidfTransformer()
    x_tfidf = tfidf_transformer.fit_transform(x_train_occurences)
    model = LinearSVC().fit(x_tfidf, y_train)

    data = usertext

    prediction = model.predict(count_vectorizer.transform([data])[0])


    if prediction[0] == 'b':
    	category = "Business"
    	context = {'result': category}
    elif prediction[0] == 't':
    	category = "Science"
    	context = {'result': category}
    elif prediction[0] == 'e':
    	category = "Entatainment"
    	context = {'result': category}
    elif prediction[0] == 'm':
    	category = "Health"
    	context = {'result': category}
    else:
    	category = "None"
    	context = {'result': category}

    print("Category is ", context)	

    return render(request, 'bonge/result.html', {'context': context})


def team(request):
    return render(request, 'bonge/team.html')

def predicto(request):
    return render(request, 'bonge/result.html')

def handler404(request, exception):
  return render(request, 'bonge/404.html')

def about(request):
  return render(request, 'bonge/about.html')


  