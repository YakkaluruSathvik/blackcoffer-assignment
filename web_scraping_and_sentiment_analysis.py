# Importing Dependencies

import requests
import html5lib
import bs4
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import chardet
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import cmudict
nltk.download('cmudict')
nltk.download('stopwords')


# root - base path (replace)
# root folder structure
# - MasterDictionary
#     - positive-words.txt
#     - negative-words.txt

# - StopWords
#     - StopWords_Auditor.txt
#     - StopWords_Currencies.txt
#     - StopWords_DatesandNumbers.txt
#     - StopWords_Generic.txt
#     - StopWords_GenericLong.txt
#     - StopWords_Geographic.txt
#     - StopWords_Names.txt

# - Input.xlsx

root = '/content/drive/MyDrive/Blackcoffer Assigment/' 
input_excel = pd.read_excel(os.path.join(root,'Input.xlsx'))

# Web scrapping and Text Extraction

Extracted_text_path = os.path.join(root,'/Extracted Text/') # Directory for extracted text files
os.makedirs(Extracted_text_path,exist_ok=True)

print("Extracting Text into files...")

def Extract_Text(header_split_pattern,footer_split_pattern,begin_clean_pattern,file_index_list,input_excel,code):

    for i in file_index_list:
        file_name = input_excel['URL_ID'][i-1]
        URL = input_excel['URL'][i-1]

        raw_html = requests.get(URL)
        if raw_html.status_code == 404:
            print(f"{URL} with {file_name} URL_ID doesn't exist")
            continue

        soup = BeautifulSoup(raw_html.content, 'html5lib')
        
        # removing unnecessary tags - script/style/footer/head
        for i in soup.find_all(['script','style','footer','head']):
            i.decompose()

        l = []
        # similarly to get all the occurrences of a given tag
        for text in soup.find('body').contents:
            txt = text.get_text()
            l.append(txt)

        parsed_txt = '.'.join(l) # list to string

        cleaned_txt = re.sub('\s+',' ',parsed_txt) # remove extra white spaces \s \n \t
        cleaned_txt.strip()

        # Split at endpoints of the main body text
        # endpoint 1 : header_split_pattern  # Before this all text is header
        # endpoint 2 : footer_split_pattern  # After this all text is footer

        header_split_text = re.split(header_split_pattern,cleaned_txt)
        if code == 1: # Both header & footer diff from common pattern
            body_text_without_header = header_split_text[1]
        else:
            body_text_without_header = header_split_text[-1]

        footer_split_text = re.split(footer_split_pattern,body_text_without_header)
        main_text = footer_split_text[0].strip() # main body text without header and footer (Article Text)

        # removing date and views which are at the
        # beginning of article text right after the title
        article_text = re.sub(begin_clean_pattern,'',main_text)

        # extracting title through h1
        article_title = soup.find('h1').text

        punctuation_title = article_title[-1]

        # Whole article text
        if punctuation_title == '.' or punctuation_title == '?' or punctuation_title == '!':
            whole_text = article_title + article_text
        else:
            whole_text = article_title + '.' + article_text

        # Specify the file path
        file_path = os.path.join(Extracted_text_path,f"{file_name}.txt")

        # Open the file in write mode ('w')
        with open(file_path, 'w') as file:
            # Write the text to the file
            file.write(whole_text)

# code == 0 : General pattern/Header different from general/Footer different from general
# code == 1 : Both Header and Footer different from general

# General or common pattern

file_indx_list = [i for i in range(1,len(input_excel)+1)]
header_split_pattern = 'By Ajay Bidyarthy -'
footer_split_pattern = 'Blackcoffer Insights'
begin_clean_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\s+\d+'

Extract_Text(header_split_pattern,footer_split_pattern,begin_clean_pattern,file_indx_list,input_excel,0)

# For the URLs with URL_ID's - 14,20,29,43,92,99,100
# Headers are different from common pattern

headers_not_trimmed = [14,20,29,43,92,99,100]
header_split_pattern = 'By Ajay Bidyarthy'
footer_split_pattern = 'Blackcoffer Insights'
begin_clean_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\s+\d+\s+\d+\s+Share FacebookTwitterPinterestWhatsApp'

Extract_Text(header_split_pattern,footer_split_pattern,begin_clean_pattern,headers_not_trimmed,input_excel,0)

# For the URLs with URL_ID's - 46,47,93
# Footers are different from common pattern

footers_not_trimmed = [46,47,93]
header_split_pattern = 'By Ajay Bidyarthy -'
footer_split_pattern = 'RELATED ARTICLESMORE'
begin_clean_pattern = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\s+\d+'

Extract_Text(header_split_pattern,footer_split_pattern,begin_clean_pattern,footers_not_trimmed,input_excel,0)

# For the URLs with URL_ID's - 83,84
# Both Headers & Footers are different from common pattern

headers_and_footers_not_trimmed = [83,84]
header_split_pattern = 'Share FacebookTwitterPinterestWhatsApp'
footer_split_pattern = 'Share FacebookTwitterPinterestWhatsApp'
begin_clean_pattern = r''

Extract_Text(header_split_pattern,footer_split_pattern,begin_clean_pattern,headers_and_footers_not_trimmed,input_excel,1)

print("Text Extraction completed.")

# Sentiment Analysis

print("Starting Text Analysis...")

MasterDictionary = os.path.join(root,'MasterDictionary/')
StopWords_path = os.path.join(root,'StopWords/')

positive_words_list_path = os.path.join(MasterDictionary,'positive-words.txt')
negative_words_list_path = os.path.join(MasterDictionary,'negative-words.txt')

positive = set()
negative = set()

def make_set(file_path,set_name):
    # Open the file in read mode ('r')
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']

    with open(file_path, 'r', encoding=encoding) as f:
        for word_newline in f:
            word_newline = word_newline.lower()
            word = re.sub('\n','',word_newline)
            set_name.add(word)

make_set(positive_words_list_path,positive)

make_set(negative_words_list_path,negative)

def find_stopWords(stopwords_path):
    set_stopWords = set()

    for stop_word_file in os.listdir(stopwords_path):
        stop_word_file_path = os.path.join(stopwords_path,stop_word_file)
        # print(stop_word_file)

        with open(stop_word_file_path, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']

        with open(stop_word_file_path, 'r', encoding=encoding) as f:
            for word_newline in f:
                word_newline = re.split('\|',word_newline)[0].strip()
                word_newline = word_newline.lower()
                word = re.sub('\n','',word_newline)
                set_stopWords.add(word)

    return set_stopWords

set_stopWords = find_stopWords(StopWords_path)

positive_dictionary = positive.difference(positive.intersection(set_stopWords))

negative_dictionary = negative.difference(negative.intersection(set_stopWords))

"""Word and Sentence Tokens"""

def word_tokenize(text):
    # Removing punctuations by splitting
    # at any of below characters .,!&?;:\-\s()
    pattern = r"[.,!&?;:\-\s()\"]+"

    # Tokenize the text
    word_tokens = re.split(pattern, text)

    # Remove empty tokens
    word_tokens = [token for token in word_tokens if token]
    return word_tokens

"""Positive and Negative score"""

def pos_and_neg_score(tokens_list):
    positive_score,negative_score = 0,0
    for token in tokens_list:
        if token.lower() in positive_dictionary:
            positive_score += 1

        elif token.lower() in negative_dictionary:
            negative_score -= 1

    negative_score *= -1
    return positive_score,negative_score

"""Polarity score"""

def polarity_score(positive_score,negative_score):
    polarity_Score = (positive_score - negative_score)/ ((positive_score + negative_score) + 0.000001)
    return polarity_Score

"""Word count (cleaned words)"""

def clean_words(word_tokens):
    from nltk.corpus import stopwords
    stopwords_nltk = stopwords.words('english')

    # Punctuations already removed
    # Stopword removal using NLTK
    words_cleaned = [w for w in word_tokens if (w not in stopwords_nltk)]

    words_count = len(words_cleaned)
    return words_count

"""Subjectivity score"""

def subjectivity_score(positive_score,negative_score,words_count):
    subjectivity_Score = (positive_score + negative_score)/(words_count + 0.000001)
    return subjectivity_Score

"""Average sentence length / Average number of words per sentence"""

def average_sentence_length(word_tokens_count,sent_count):
    average_sent_length = word_tokens_count/sent_count
    return average_sent_length

"""Syllabe count & Complex words"""

syllable_dict = cmudict.dict()

def count_syllables(word):
    if word.lower() not in syllable_dict:    # search for lower case version of the word in dictionary
        return 0
    return [len(list(y for y in x if y[-1].isdigit())) for x in syllable_dict[word.lower()]][0]
                                               # return number of syllable

def is_complex(word):
    syllable_count = count_syllables(word)
    return syllable_count > 2

def count_complex_words(words):
    total_syllable = 0
    for word in words:
        word = word.lower()
        total_syllable += count_syllables(word)

    num_complex_words = sum(is_complex(word) for word in words)
    return total_syllable,num_complex_words

"""Percentage of Complex words"""

def percent_complex_words(complex_word_count,word_tokens_count):
    percent_of_complex_words = complex_word_count/word_tokens_count
    return percent_of_complex_words

"""Fog index"""

def fog_index(average_sentence_length,percent_of_complex_words):
    Fog_index = 0.4*(average_sentence_length + percent_of_complex_words)
    return Fog_index

"""Personal Pronouns"""

def count_personal_pronouns(word_tokens):
    total_personal_pronouns = 0
    for word in word_tokens:
        pattern = r"\b(I|we|my|ours|us)\b"   # pattern to check if those words exists
        pattern = r"(?<!\bUS\b)" + pattern   # pattern should not include US instead of us
        matches = re.findall(pattern, word, flags=re.IGNORECASE)
        total_personal_pronouns += len(matches)
    return total_personal_pronouns

"""Average Word Length"""

def average_word_length(word_tokens):
    word_length = 0

    for word in word_tokens:
        word_length += len(word)

    avg_word_length = word_length/len(word_tokens)

    return avg_word_length

cols = ['URL_ID',
        'URL',
        'POSITIVE SCORE',
        'NEGATIVE SCORE',
        'POLARITY SCORE',
        'SUBJECTIVITY SCORE',
        'AVG SENTENCE LENGTH',
        'PERCENTAGE OF COMPLEX WORDS',
        'FOG INDEX',
        'AVG NUMBER OF WORDS PER SENTENCE',
        'COMPLEX WORD COUNT',
        'WORD COUNT',
        'SYLLABLE PER WORD',
        'PERSONAL PRONOUNS',
        'AVG WORD LENGTH']

output = pd.DataFrame(columns=cols)

output['URL_ID'] = input_excel['URL_ID']
output['URL'] = input_excel['URL']

modified_cols = ['POSITIVE SCORE',
                'NEGATIVE SCORE',
                'POLARITY SCORE',
                'SUBJECTIVITY SCORE',
                'AVG SENTENCE LENGTH',
                'PERCENTAGE OF COMPLEX WORDS',
                'FOG INDEX',
                'AVG NUMBER OF WORDS PER SENTENCE',
                'COMPLEX WORD COUNT',
                'WORD COUNT',
                'SYLLABLE PER WORD',
                'PERSONAL PRONOUNS',
                'AVG WORD LENGTH']

syllable_dict = cmudict.dict()

for i in range(len(input_excel)):
    file_name = input_excel['URL_ID'][i]
    URL = input_excel['URL'][i]

    raw_html = requests.get(URL)
    if raw_html.status_code == 404:
        print(f"{URL} with {file_name} URL_ID doesn't exist")
        continue

    text_file_path = os.path.join(Extracted_text_path,file_name+'.txt')

    with open(text_file_path, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']

    # Open the file in read mode ('r')
    with open(text_file_path,'r',encoding=encoding) as file:
        text = file.read()

    # Tokenization of words and sentences
    word_tokens = word_tokenize(text)
    sent_tokens = sent_tokenize(text, language='english')

    # Count of word and sentence tokens
    sent_count = len(sent_tokens)
    word_tokens_count = len(word_tokens)

    # Count of cleaned words (after removal of stopwords and punctuations)
    words_count = clean_words(word_tokens)

    # Positive and Negative Scores
    positive_Score,negative_Score = pos_and_neg_score(word_tokens)

    # Polarity Score
    polarity_Score = round(polarity_score(positive_Score,negative_Score),4)

    # Subjectivity Score
    subjectivity_Score = round(subjectivity_score(positive_Score,negative_Score,words_count),4)

    # Average sentence length
    avg_sentence_length = round(average_sentence_length(word_tokens_count,sent_count),4)

    # Average no of words per sentence
    avg_no_of_words_per_sentence = round(average_sentence_length(word_tokens_count,sent_count),4)

    # Syllable count and Complex word count
    total_syllable_count, complex_word_count = count_complex_words(word_tokens)

    # Percentage of complex words
    percent_of_complex_words = round(percent_complex_words(complex_word_count,word_tokens_count),4)

    # Fog Index
    fog_Index = round(fog_index(avg_sentence_length,percent_of_complex_words),4)

    # Personal Pronouns
    total_Personal_Pronouns = count_personal_pronouns(word_tokens)

    # Average word length
    avg_word_length = round(average_word_length(word_tokens),4)

    results = [positive_Score, negative_Score, polarity_Score, subjectivity_Score,
               avg_sentence_length, percent_of_complex_words, fog_Index,
               avg_no_of_words_per_sentence, complex_word_count, words_count,
               total_syllable_count, total_Personal_Pronouns, avg_word_length]

    for index in range(len(modified_cols)):
        output[modified_cols[index]][i] = results[index]

    print(f"File: {file_name+'.txt'}")
    print(f'''
              Positive Score: {positive_Score}, Negative Score: {negative_Score}, Polarity Score: {polarity_Score}, Subjectivity Score: {subjectivity_Score},
              Average Sentence Length: {avg_sentence_length}, Percentage of Complex Words: {percent_of_complex_words}, Fog Index: {fog_Index},
              Average No of Words per Sentence:{avg_no_of_words_per_sentence}, Complex Word Count: {complex_word_count}, Word Count: {words_count},
              Syllable count: {total_syllable_count}, Personal Pronouns: {total_Personal_Pronouns}, Average Word Length: {avg_word_length}
            ''')

print("Text Analysis completed.")

output_excel_path = os.path.join(root,'Output Data Structure.xlsx')
output.to_excel(output_excel_path,index=False)
print("Results saved to Output Data Structure.xlsx file")