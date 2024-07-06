import re
import os
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def cleanComment(comment):
    try:
        comment = re.sub(r'(?<=\w)-(?=\w)', 'STRIP', str(comment))
        comment = re.sub(r'http\S+|www\S+', '', str(comment))
        comment = re.sub(r'@\w+|[^\w\s-]|(?<!\w)-(?!\w)|\d+|(?<=\n)[IVXLCDM]+', ' ', str(comment))
        comment = re.sub(r'\s+', ' ', str(comment))
        comment = comment.replace('STRIP', '-')
        comment = comment.replace('\n', ' ')
        return comment.strip()
    except Exception as e:
        print(f"Err: Failed to clean comments due to {str(e)}")
        return comment

def caseFolding(comment):
    try:
        cleanComment = comment.lower()
        return cleanComment     
    except Exception as e:
        print(f"Err: Failed to case folding due to {str(e)}")
        return comment

def replaceSlangWords(comment, language):
    try:
        slang_dict = {}
        filename = f'./slang-word/slang-{language}.txt'
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'r') as f:
                for line in f:
                    slang, formal = line.strip().split(',')
                    slang_dict[slang] = formal

        words = comment.split()
        for i in range(len(words)):
            if words[i] in slang_dict:
                words[i] = slang_dict[words[i]]
        return ' '.join(words)
    except Exception as e:
        print(f"Err: Failed to replace slang words due to {str(e)}")
        return comment 

def tokenize(comment):
    try:
        words = comment.split(' ')
        words = list(filter(None, words)) 

        return words
    except Exception as e:
        print("Err: Failed to tokenize due to", str(e))
        return comment

def stopwordRemoval(comments, language):
    language_mapping = {
        'en': 'english', 
        'id': 'indonesian'
    }

    nltk_language = language_mapping.get(language, 'indonesian') 
    stopWordRemoved = []
    
    try:
        stopList = stopwords.words(nltk_language)
        filename = f'./stop-word/stopword-{language}.txt' 
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'r') as f:
                stopList.extend(f.read().split('\n')) 
        for word in comments:
            if word not in stopList:
                stopWordRemoved.append(word)

        return stopWordRemoved  
    except Exception as e:  
        print(f"Err: Failed to remove stopwords due to {str(e)}")
        return comments 

def stemming(comments, language):
    try:
        if language == 'en':
            stemmer = PorterStemmer()
        elif language == 'id':
            stemmer = StemmerFactory().create_stemmer()
        else:
            stemmer = PorterStemmer()  # Default to PorterStemmer if language not recognized

        stemmed_words = [stemmer.stem(word) for word in comments]
        return str(stemmed_words)
    except Exception as e:
        print(f"Err: Failed to stem words due to {str(e)}")
        return comments