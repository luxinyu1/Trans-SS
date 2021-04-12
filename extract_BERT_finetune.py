from access.utils.paths import DATA_DIR, DATASETS_DIR
from access.utils.calc import get_freq, get_word2freq

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

para_data = DATA_DIR / 'parabank2.tsv'
finetune_bert_data = DATASETS_DIR / 'train.tsv'

w2f = get_word2freq()
# ps = PorterStemmer()

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

complex_words = set()
for key in w2f:
    if w2f[key] < 1000:
        if wordnet_lemmatizer.lemmatize(str(key)) in w2f and w2f[wordnet_lemmatizer.lemmatize(str(key))] > 1000:
            continue
        else:
            complex_words.add(wordnet_lemmatizer.lemmatize(str(key)))

final_complex_words = []
final_synonyms = []
            
for word in complex_words:
    synsets = wn.synsets(word)
    synonyms = set()
    for syn in synsets:
        for lm in syn.lemmas():
            synonyms.add(lm.name())
    synonyms = synonyms - set([word])
    synonyms = [syn for syn in synonyms if (syn in w2f and word in w2f) and w2f[syn] > w2f[word]]

    if synonyms:
        final_complex_words.append(word)
        final_synonyms.append(synonyms)
        
assert len(final_complex_words) == len(final_synonyms)

lenfile = sum([1 for i in open(para_data, "r")])

with open(para_data, 'r', encoding='utf-8') as f:
    with open(finetune_bert_data, 'w', encoding='utf-8') as f_finetune:
        for line in tqdm(f, total=lenfile):
            tabs = line.strip().split('\t')
            sentences = tabs[1:]
            origin_sentence = sentences[0]
            origin_tokens = word_tokenize(origin_sentence.lower())
            lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in origin_tokens]
            complex_tokens = list(set(final_complex_words) & set(lemmatized_tokens))
            possible_synonyms = [final_synonyms[final_complex_words.index(token)] for token in complex_tokens]
            sub_sentences = sentences[1:]
            for sub_sentence in sub_sentences:
                sub_tokens = word_tokenize(sub_sentence.lower())
                lemmatized_sub_tokens = [wordnet_lemmatizer.lemmatize(token) for token in sub_tokens]
                for e in possible_synonyms:
                    for w in e:
                        if w in lemmatized_sub_tokens:
                            f_finetune.write(origin_sentence+"\t")
                            f_finetune.write(sub_sentence+"\t")
                            f_finetune.write(origin_tokens[lemmatized_tokens.index(complex_tokens[possible_synonyms.index(e)])]+"\t")
                            f_finetune.write(sub_tokens[lemmatized_sub_tokens.index(w)]+"\n")

#             for sentence in sub_sentences:
#                 tokens = word_tokenize(sentence.lower())
#                 diff_a = list(set(tokens)-set(origin_tokens))
#                 diff_b = list(set(origin_tokens)-set(tokens))
#                 if len(diff_a)==1 and len(diff_b)==1:
#                     new = ps.stem(diff_a[0].lower())
#                     origin = ps.stem(diff_b[0].lower())
#                     if new not in w2f or origin not in w2f:
#                         continue
#                     if new == origin:
#                         continue
#                     else:
#                         synsets = wn.synsets(origin)
#                         for item in synsets:
#                             names = item.lemma_names()
#                             names = list(set(names) - set(origin))
#                             if new in names:
#                                 origin_freq = w2f[origin]
#                                 new_freq = w2f[new]
#                                 if new_freq < origin_freq:
#                                     f_finetune.write(sentence+'\t')
#                                     f_finetune.write(origin_sentence+'\t')
#                                     f_finetune.write(diff_a[0]+'\t')
#                                     f_finetune.write(diff_b[0]+'\n')
#                                 else:
#                                     f_finetune.write(origin_sentence+'\t')
#                                     f_finetune.write(sentence+'\t')
#                                     f_finetune.write(diff_b[0]+'\t')
#                                     f_finetune.write(diff_a[0]+'\n')
#                                 break

            