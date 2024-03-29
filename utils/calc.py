import re
from nltk.corpus import cmudict, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from functools import lru_cache
import math
import torch

from utils.paths import DATA_DIR
from .utils import read_lines

d = cmudict.dict()

@lru_cache(maxsize=100000)
def nsyl(word):
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
    except:
        return syllables(word)

@lru_cache(maxsize=100000)
def syllables(word):
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return [count]

def is_punctuation(token):
    return re.match('^[.,\/#!$%\'\^&\*;:{}=\-_`~()]$', token) is not None

def is_blank_space(token):
    return re.match('\s+', token) is not None

def sentence_fkgl(sentence):
    sent_tokenized = word_tokenize(sentence)
    sent_num = len(sent_tokenize(sentence))
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    _words_in_sent = len(words_in_sent)
    syllables_all = 0
    for word in words_in_sent:
        syllables_all += syllables(word)[0]
    avg_syllables_per_word = syllables_all / _words_in_sent
    avg_words_per_sentence = _words_in_sent / sent_num
    fkgl = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
    return fkgl

def count_words_in_sentence(sentence):
    sent_tokenized = word_tokenize(sentence)
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    return len(words_in_sent)

def count_words_in_sentence_fr(sentence):
    sent_tokenized = word_tokenize(sentence, language="french")
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    return len(words_in_sent)

def count_words_in_sentence_es(sentence):
    sent_tokenized = word_tokenize(sentence, language="spanish")
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    return len(words_in_sent)

def sentence_fkf(sentence):
    sent_num = len(sent_tokenize(sentence))
    sent_tokenized = word_tokenize(sentence)
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    _words_in_sent = len(words_in_sent)
    syllables_all = 0
    for word in words_in_sent:
        syllables_all += syllables(word)[0]
    avg_syllables_per_word = syllables_all / _words_in_sent
    avg_words_per_sentence = _words_in_sent / sent_num
    fkf = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
    return fkf

@lru_cache(maxsize=100000)
def syllables_de(word):
    # every syllable either has a vowel or a diphtong (ai/au/ei/eu)
    num_syllables = 0
    last = ''
    word = word.lower()
    diphtongs = [
        'au','ei','ai','eu','äu','ie','aa','ee','ui'
    ]
    for char in word:
        if char in "aeiouäüö":
            num_syllables += 1
            if last:
                if last+char in diphtongs:
                    num_syllables -= 1
            last = char
        else:
            last = ''
    
    return num_syllables
            
def sentence_fkf_de(sentence):
    sent_num = len(sent_tokenize(sentence, language="german"))
    sent_tokenized = word_tokenize(sentence, language="german")
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    _words_in_sent = len(words_in_sent)
    syllables_all = 0
    for word in words_in_sent:
        syllables_all += syllables_de(word)
    avg_syllables_per_word = syllables_all / _words_in_sent
    avg_words_per_sentence = _words_in_sent / sent_num
    fkf = 180 - 58.5*avg_syllables_per_word - avg_words_per_sentence
    return fkf

@lru_cache(maxsize=100000)
def syllables_fr(word):
    # every syllable either has a vowel or a diphtong
    num_syllables = 0
    word = word.lower()
    last = ''
    diphtongs = [
        'ei','ai','au','ou','oû','eu','æu'
    ]
    for char in word:
        if char in "iîïaàâeéèêuûoôy":
            num_syllables += 1
            if last:
                if last+char in diphtongs:
                    num_syllables -= 1
            last = char
        else:
            last = ''
    
    if word.endswith('e'):
        num_syllables -= 1
    if word.endswith('le'):
        num_syllables += 1
    if num_syllables == 0:
        num_syllables += 1
            
    return num_syllables

def sentence_fkf_fr(sentence):
    sent_num = len(sent_tokenize(sentence, language="french"))
    sent_tokenized = word_tokenize(sentence, language="french")
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    _words_in_sent = len(words_in_sent)
    syllables_all = 0
    for word in words_in_sent:
        syllables_all += syllables_fr(word)
    avg_syllables_per_word = syllables_all / _words_in_sent
    avg_words_per_sentence = _words_in_sent / sent_num
    fkf = 207.0 - 73.6*avg_syllables_per_word - 1.015*avg_words_per_sentence
    return fkf

def corpus_fkf_fr(corpus):
    corpus = ' '.join(corpus)
    return sentence_fkf_fr(corpus)

@lru_cache(maxsize=100000)
def syllables_es(word):
    # every syllable either has a vowel or a diphtong
    num_syllables = 0
    word = word.lower()
    last = ''
    diphtongs = [
        'ai', 'ei','oi','au','eu','ou','ui','ia','ie','io','iu','ua','ue','uo'
    ]
    for char in word:
        if char in "iaeéuoy":
            num_syllables += 1
            if last:
                if last+char in diphtongs:
                    num_syllables -= 1
            last = char
        else:
            last = ''
            
    return num_syllables

def sentence_fkf_es(sentence):
    sent_num = len(sent_tokenize(sentence, language="spanish"))
    sent_tokenized = word_tokenize(sentence, language="spanish")
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    _words_in_sent = len(words_in_sent)
    syllables_all = 0
    for word in words_in_sent:
        syllables_all += syllables_es(word)
    avg_syllables_per_word = syllables_all / _words_in_sent
    avg_words_per_sentence = _words_in_sent / sent_num
    fkf = 206.835 - 60.0*avg_syllables_per_word - 1.02*avg_words_per_sentence
    return fkf

def sentence_gfi(sentence):
    sent_num = len(sent_tokenize(sentence))
    sent_tokenized = word_tokenize(sentence)
    _is_long_word = lambda w: len(w) > 7 # GFI defines that "longWords" are words longer than 7 characters
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    _words_in_sent = len(words_in_sent)
    long_words_in_sent = [w for w in words_in_sent if _is_long_word(w)]
    _long_words_in_sent = len(long_words_in_sent)
    return 0.4 * (_words_in_sent/sent_num + 100*_long_words_in_sent/_words_in_sent)

def sentence_ari(sentence):
    sent_num = len(sent_tokenize(sentence))
    characters_in_sent = [ch for ch in sentence if not is_punctuation(ch) and not is_blank_space(ch)]
    _characters_in_sent = len(characters_in_sent)
    sent_tokenized = word_tokenize(sentence)
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    _words_in_sent = len(words_in_sent)
    return 4.71 * (_characters_in_sent/_words_in_sent) + 0.5 * (_words_in_sent/sent_num) - 21.43

def sentence_dcrf(sentence):
    dc3000 = read_lines(DATA_DIR / 'dc_3000.txt')
    _is_difficult = lambda w: w.lower() not in dc3000
    sent_num = len(sent_tokenize(sentence))
    sent_tokenized = word_tokenize(sentence)
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    _words_in_sent = len(words_in_sent)
    diff_words = [w for w in words_in_sent if _is_difficult(w)]
    _diff_words = len(diff_words)
    dcrf = 0.1579*(_diff_words/_words_in_sent*100) + 0.0496*(_words_in_sent/sent_num)
    if _diff_words/_words_in_sent > 0.05:
        dcrf += 3.6365
    return dcrf

def sentence_smog(sentence):
    # "numberOfPolysyllables" is the number of words with three or more syllables
    sent_num = len(sent_tokenize(sentence))
    sent_tokenized = word_tokenize(sentence)
    words_in_sent = [token for token in sent_tokenized if not is_punctuation(token)]
    poly_syllables = [w for w in words_in_sent if syllables(w)[0] >= 3]
    _poly_syllables = len(poly_syllables)
    return 1.0430*math.sqrt(_poly_syllables*30/sent_num) + 3.1291

def corpus_fkf_es(corpus):
    corpus = ' '.join(corpus)
    return sentence_fkf_es(corpus)

def ppl(model, tokenizer, sentence):
    # It is very rare that we encounter sentence's length greater than the fixed max_seq_length of models,
    # so here we don't apply the sliding-window strategy
    encodings = tokenizer(sentence, return_tensors='pt')
    input_ids = encodings["input_ids"].to("cuda")
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids)[0]
        ppl = math.exp(loss)
    return ppl

def get_corpus_vocab_size(corpus):
    words = []
    for line in corpus:
        word_tokens = word_tokenize(line, language="english")
        for word in word_tokens:
            _word = word.lower()
            if _word not in words and re.match(r"[a-zA-Z]+", word):
                words.append(_word)
    stop_words = set(stopwords.words('english'))
    vocabs = set(words) - set(stop_words)
    return len(vocabs)

def get_corpus_vocab_size_fr(corpus):
    words = []
    for line in corpus:
        word_tokens = word_tokenize(line, language="french")
        for word in word_tokens:
            if word not in words and re.match(r"[a-zA-ZÀ-ÿ]+", word):
                words.append(word)
    stop_words = set(stopwords.words('french'))
    vocabs = [w for w in words if w not in stop_words]
    return len(vocabs)

def get_corpus_vocab_size_es(corpus):
    words = []
    for line in corpus:
        word_tokens = word_tokenize(line, language="spanish")
        for word in word_tokens:
            if word not in words and re.match(r"[a-zA-ZÀ-ÿ]+", word):
                words.append(word)
    stop_words = set(stopwords.words('spanish'))
    vocabs = [w for w in words if w not in stop_words]
    return len(vocabs)

def get_avg_token_per_sentence(corpus):
    total_line = 0
    total_words = 0
    stop_words = set(stopwords.words('english'))
    for line in corpus:
        total_line += 1
        total_words += count_words_in_sentence(line)
    return total_words*1.0 / total_line*1.0

def get_avg_token_per_sentence_fr(corpus):
    total_line = 0
    total_words = 0
    stop_words = set(stopwords.words('french'))
    for line in corpus:
        total_line += 1
        total_words += count_words_in_sentence_fr(line)
    return total_words*1.0 / total_line*1.0

def get_avg_token_per_sentence_es(corpus):
    total_line = 0
    total_words = 0
    stop_words = set(stopwords.words('spanish'))
    for line in corpus:
        total_line += 1
        total_words += count_words_in_sentence_es(line)
    return total_words*1.0 / total_line*1.0