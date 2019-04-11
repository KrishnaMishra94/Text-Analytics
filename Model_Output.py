import pandas as pd
################# DESERIALIZING PREVIOUSLY CREATED MODEL ##################
import pickle
file = open('US_COMPANIES_CLASSIFICATION.pkl', 'rb')
model = pickle.load(file)
file.close()
##########################################################################


################ READING THE TEST FILE ##############################

data = pd.read_excel('Company_ Business Description.xlsx')
data_orig = pd.read_excel('Company_ Business Description.xlsx')   
data = data[['Business Description']]
data = data[data['Business Description'] != '-']

######################################################################


############################## CLEANING TEXT ##################################
import re

def clean_text(text):
    ## REMOVING THE CHARACTERS [\], ['] and ["]
    text = re.sub(r"\\"," ",text)
    text = re.sub(r"\""," ",text)
    text = re.sub(r"\'"," ",text)
    
    ## REMOVING SINGLE CHARACTERS
    text = re.sub(r"\s+[a-zA-Z]\s+"," ",text)
    
    ## REMOVING MULTIPLE SPACES WITH SINGLE SPACE
    text = re.sub(r"\s+"," ",text)

    ## CONVERTING TEXT TO LOWERCASE    
    text = text.strip().lower()
    
    ## REPLACING PUNCTUATION CHARACTERS WITH SPACES
    punctuations    = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict  = dict((char, " ") for char in punctuations)
    translate_map   = str.maketrans(translate_dict)
    text            = text.translate(translate_map)
    
    return text

data['Business Description'] = data['Business Description'].apply(lambda var : clean_text(var))

##############################################################################

######################### PERFORMING LEMMATIZATION ##########################
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lammetize_text(txt):
    words = word_tokenize(txt)
    val = [lemmatizer.lemmatize(w) for w in words]
    seperator = ' '
    return(seperator.join(val))
    
data['Business Description'] = data['Business Description'].apply(lambda var : lammetize_text(var))

##############################################################################


######################## PERFORMING BAG OF WORDS MATRIX #####################

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words="english",max_features=15000)
X_vectors   = count_vect.fit_transform(data['Business Description']).toarray()

##########################################################################


predict = pd.Series(model.predict(X_vectors))

data_orig['Industry Classifications'] = predict
data_orig.to_excel('Company_Business_Description_Output.xlsx')

#
    

