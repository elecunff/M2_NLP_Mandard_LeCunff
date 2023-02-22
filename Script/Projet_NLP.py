
import json
import pandas as pd
import re
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords
import spacy

liste = ['formation0', 'formation1', 'formation2', 'formation3', 'formation4',
         'rupture0', 'rupture1', 'rupture2', 'rupture3', 'rupture4']

date = []
texte = []
theme = []

for obj in liste :
    f = open(f"C:/Users/Mael/Documents/Fac/M2/NLP/M2_NLP_Mandard_LeCunff/Data/{obj}.json", encoding= 'UTF-8')
    #f = open(f"C:/Users/ewen/Documents/M2/NLP/M2_NLP_Mandard_LeCunff/Data/{obj}.json", encoding= 'UTF-8')
    
    data = json.load(f)
    
    them = data['query']['theme']
    
    results = data['results']
    
    for res in results : 
        theme.append(them[0])
        date.append(res['decision_date'])
        texte.append(res['text'])
    

fin_data = pd.DataFrame({'date' : date, 'theme' : theme, 'texte' : texte})

pd.Series(date).nunique()
pd.Series(texte).nunique()


#1ere etape : mettre en minuscule
#2eme etape : enlever et tokenisation
#3eme etape : stop words
#4eme etape : lemmatisation

def mise_en_minuscule(text):
    return text.replace(text, text.lower())

def remove_accents(text) : 
    return unidecode(text)

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text) #supprimer les caractères non alphanumériques
    text = re.sub(r'\b\d+\b', '', text) #supprimer les mots qui contiennent un chiffre
    text = re.sub(r'\bxx+\b|\bXX+\b', '', text) #supprimer les mots qui contiennent une chaîne de "xx" ou "XX"
    return text



fin_data['texte'] = fin_data['texte'].apply(mise_en_minuscule)

fin_data['texte'] = fin_data['texte'].apply(lambda x: remove_accents(x))

fin_data['texte'] = fin_data['texte'].apply(lambda x: clean_text(x))


nltk.download('stopwords')
stop_words = set(stopwords.words('french'))
liste_stop = ['b', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'soc', 'cds', 'r', 'p', 'u', 'q', 
              'peuple', 'fs', 'audience', 'publique']

stop_words.update(liste_stop)

def supprime_stopwords(text):
    words = re.split(r"[ ]", text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

fin_data['texte'] = fin_data['texte'].apply(supprime_stopwords)


def supprime_espace(text):
    words = re.split(r"[ ]", text)
    filtered_words = [word for word in words if word]
    return ' '.join(filtered_words)

fin_data['texte'] = fin_data['texte'].apply(supprime_espace)





# python -m spacy download fr_core_news_sm
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])


def lemmatize(allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    fin_data['texte'] = fin_data['texte'].apply(lambda x: " ".join([token.lemma_ for token in nlp(x) if token.pos_ in allowed_postags]))
    return fin_data['texte']

fin_data['texte'] = lemmatize()


print(fin_data['texte'][35])

#fin_data['texte'][1].split()[:54]













































"""
from  spacy.lang.fr.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
print(stopwords)


from nltk.stem import WordNetLemmatizer
  
import nltk
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

print("rocks :", lemmatizer.lemmatize("etait"))
print("corpora :", lemmatizer.lemmatize("corpora"))


# Import libraries for text manipulation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

#vec = CountVectorizer(decode_error='ignore', stop_words='english', max_features=500, lowercase=True, 
#    min_df=20, encoding='utf-8')
vec = TfidfVectorizer(decode_error='ignore', stop_words='english', max_features=500, lowercase=True, 
                      min_df=20, encoding='utf-8', strip_accents='unicode')
# max_features : top max_features ordered by term frequency across the corpus
X = vec.fit_transform(fin_data['texte'])

vec.get_feature_names_out()

# X sparse doit être transformé en matrice non sparse pour les traitements ultérieurs
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
print(df.head(4))
print(df.shape)


# importer le vectoriseur TfidfVectorizer de Scikit-Learn.  
from sklearn.feature_extraction.text import TfidfVectorizer

vectoriseur = TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True, norm=None)
documents_transformes = vectoriseur.fit_transform(fin_data['texte'])


documents_transformes_tableau = documents_transformes.toarray()
# la prochaine ligne de code vérifie que le tableau numpy contient le même nombre
# de documents que notre liste de fichiers
len(documents_transformes_tableau)

vectoriseur.get_feature_names_out()

"""


from wordcloud import WordCloud

wordcloud = WordCloud(background_color="white")

text = " ".join(fin_data['texte'].tolist()) 

wordcloud.generate(text)

import matplotlib.pyplot as plt

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()

