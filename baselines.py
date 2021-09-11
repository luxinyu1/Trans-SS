from utils.utils import get_data_filepath, get_dataset_dir, read_lines
from easse.report import get_all_scores
from easse.sari import corpus_sari

from utils.calc import corpus_fkf_fr, corpus_fkf_es
import nltk
import random
import numpy as np

BTTS10_PATH = './baseline_sys_outputs/BTTS10'
MUSS_es_PATH = './baseline_sys_outputs/MUSS-es'
MUSS_fr_PATH = './baseline_sys_outputs/MUSS-fr'

# def truncate(row_sentence, ratio, lang="english"):
#     tokenized_sentence = nltk.word_tokenize(row_sentence, language=lang)
#     target_words_in_sent = int(len(tokenized_sentence)*0.8)
#     target_words = tokenized_sentence[:target_words_in_sent:]
#     return " ".join(target_words)

turk_complex_path = get_data_filepath('turkcorpus', 'test', 'complex')
turk_ref_paths = [get_data_filepath('turkcorpus', 'test', 'simple.turk', i) for i in range(8)]

asset_complex_path = get_data_filepath('turkcorpus', 'test', 'complex') # same 2000 sentences
asset_ref_paths = [get_data_filepath('asset', 'test', 'simp', i) for i in range(10)]

newsela_complex_path = get_data_filepath('newsela', 'test', 'src')
newsela_ref_paths = [get_data_filepath('newsela', 'test', 'dst')]

alector_complex_path = get_data_filepath('alector', 'test', 'complex')
alector_ref_paths = [get_data_filepath('alector', 'test', 'simple')]

simplext_complex_path = get_data_filepath('simplext', 'test', 'complex')
simplext_ref_paths = [get_data_filepath('simplext', 'test', 'simple')]

print("-"*10 + "Identity Baseline" + "-"*10)

turk_scores = get_all_scores(orig_sents=read_lines(turk_complex_path), sys_sents=read_lines(turk_complex_path), refs_sents=[read_lines(turk_ref_path) for turk_ref_path in turk_ref_paths], lowercase=True)
print(turk_scores)

asset_scores = get_all_scores(orig_sents=read_lines(asset_complex_path), sys_sents=read_lines(asset_complex_path), refs_sents=[read_lines(asset_ref_path) for asset_ref_path in asset_ref_paths], lowercase=True)
print(asset_scores)

newsela_scores = get_all_scores(orig_sents=read_lines(newsela_complex_path), sys_sents=read_lines(newsela_complex_path), refs_sents=[read_lines(newsela_ref_path) for newsela_ref_path in newsela_ref_paths], lowercase=True)
print(newsela_scores)

alector_scores = get_all_scores(orig_sents=read_lines(alector_complex_path), sys_sents=read_lines(alector_complex_path), refs_sents=[read_lines(alector_ref_path) for alector_ref_path in alector_ref_paths], lowercase=True)
print(alector_scores)

alector_fr_fres = corpus_fkf_fr(read_lines(alector_complex_path))
print("[ALECTOR]", alector_fr_fres)

simplext_scores = get_all_scores(orig_sents=read_lines(simplext_complex_path), sys_sents=read_lines(simplext_complex_path), refs_sents=[read_lines(simplext_ref_path) for simplext_ref_path in simplext_ref_paths], lowercase=True)
print(simplext_scores)

simplext_es_fres = corpus_fkf_es(read_lines(simplext_complex_path))
print("[SIMPLEXT]", simplext_es_fres)

print("-"*10 + "Pivot Baseline" + "-"*10)

FRENCH_PIVOT = './baseline_sys_outputs/alector_output_pivot.txt'

alector_scores = get_all_scores(orig_sents=read_lines(alector_complex_path), sys_sents=read_lines(FRENCH_PIVOT), refs_sents=[read_lines(alector_ref_path) for alector_ref_path in alector_ref_paths], lowercase=True)

pivot_fr_fres = corpus_fkf_fr(read_lines(FRENCH_PIVOT))

print("[PIVOT ALECTOR]", pivot_fr_fres)

print(alector_scores)

SPANISH_PIVOT = './baseline_sys_outputs/simplext_output_pivot.txt'

simplext_scores = get_all_scores(orig_sents=read_lines(simplext_complex_path), sys_sents=read_lines(SPANISH_PIVOT), refs_sents=[read_lines(simplext_ref_path) for simplext_ref_path in simplext_ref_paths], lowercase=True)

pivot_es_fres = corpus_fkf_es(read_lines(SPANISH_PIVOT))

print("[PIVOT SIMLEXT]", pivot_es_fres)

print(simplext_scores)

# print("-"*10 + "Truncation Baseline" + "-"*10)

# turk_complex_sentences = read_lines(turk_complex_path)
# turk_simple_sentences = [truncate(line, 0.8) for line in turk_complex_sentences]
# turk_scores = get_all_scores(orig_sents=turk_complex_sentences, sys_sents=turk_simple_sentences, refs_sents=[read_lines(turk_ref_path) for turk_ref_path in turk_ref_paths], lowercase=True)
# print(turk_scores)

# asset_complex_sentences = read_lines(asset_complex_path)
# asset_simple_sentences = [truncate(line, 0.8) for line in asset_complex_sentences]
# asset_scores = get_all_scores(orig_sents=asset_complex_sentences, sys_sents=asset_simple_sentences, refs_sents=[read_lines(asset_ref_path) for asset_ref_path in asset_ref_paths], lowercase=True)
# print(asset_scores)

# alector_complex_sentences = read_lines(alector_complex_path)
# alector_simple_sentences = [truncate(line, 0.8) for line in alector_complex_sentences]
# alector_scores = get_all_scores(orig_sents=alector_complex_sentences, sys_sents=alector_simple_sentences, refs_sents=[read_lines(alector_ref_path) for alector_ref_path in alector_ref_paths], lowercase=True)
# print(alector_scores)

# simplext_complex_sentences = read_lines(simplext_complex_path)
# simplext_simple_sentences = [truncate(line, 0.8) for line in simplext_complex_sentences]
# simplext_scores = get_all_scores(orig_sents=simplext_complex_sentences, sys_sents=simplext_simple_sentences, refs_sents=[read_lines(simplext_ref_path ) for simplext_ref_path in simplext_ref_paths], lowercase=True)
# print(simplext_scores)

# newsela_complex_sentences = read_lines(newsela_complex_path)
# newsela_simple_sentences = [truncate(line, 0.8) for line in newsela_complex_sentences]
# newsela_scores = get_all_scores(orig_sents=newsela_complex_sentences, sys_sents=newsela_simple_sentences, refs_sents=[read_lines(newsela_ref_path) for newsela_ref_path in newsela_ref_paths], lowercase=True)
# print(newsela_scores)

print("-"*10 + "Reference Baseline" + "-"*10)

sari_ref_scores = []
fkgl_ref_scores = []
bleu_ref_scores = []
compression_ratios = []

for i in range(len(turk_ref_paths)):
    _paths = turk_ref_paths[0:i] + turk_ref_paths[i+1:]
    _duplicate = random.sample(turk_ref_paths, 1)[0]
    _paths.append(_duplicate)
    ref_score = get_all_scores(orig_sents=read_lines(turk_complex_path), sys_sents=read_lines(turk_ref_paths[i]), refs_sents=[read_lines(p) for p in _paths], lowercase=True)
    sari_ref_scores.append(ref_score["SARI"])
    fkgl_ref_scores.append(ref_score["FKGL"])
    bleu_ref_scores.append(ref_score["BLEU"])
    compression_ratios.append(ref_score["Compression ratio"])

print("[turk] reference bleu:", np.mean(bleu_ref_scores))
print("[turk] reference sari:", np.mean(sari_ref_scores))
print("[turk] reference fkgl:", np.mean(fkgl_ref_scores))
print("[turk] reference compression ratio", np.mean(compression_ratios))

sari_ref_scores = []
fkgl_ref_scores = []
bleu_ref_scores = []
compression_ratios = []

for i in range(len(asset_ref_paths)):
    _paths = asset_ref_paths[0:i] + asset_ref_paths[i+1:]
    _duplicate = random.sample(turk_ref_paths, 1)[0]
    _paths.append(_duplicate)
    ref_score = get_all_scores(orig_sents=read_lines(asset_complex_path), sys_sents=read_lines(asset_ref_paths[i]), refs_sents=[read_lines(p) for p in _paths], lowercase=True)
    sari_ref_scores.append(ref_score["SARI"])
    fkgl_ref_scores.append(ref_score["FKGL"])
    bleu_ref_scores.append(ref_score["BLEU"])
    compression_ratios.append(ref_score["Compression ratio"])
    
print("[asset] reference bleu:", np.mean(bleu_ref_scores))
print("[asset] reference sari:", np.mean(sari_ref_scores))
print("[asset] reference fkgl:", np.mean(fkgl_ref_scores))
print("[asset] reference compression ratio:", np.mean(compression_ratios))

# alector_scores = get_all_scores(orig_sents=read_lines(alector_complex_path), sys_sents=read_lines(alector_ref_paths[0]), refs_sents=[read_lines(alector_ref_path) for alector_ref_path in alector_ref_paths], lowercase=True)

# print("[alector]:", alector_scores)

# alector_sari = corpus_sari(orig_sents=read_lines(alector_complex_path), sys_sents=read_lines(alector_ref_paths[0]), refs_sents=[read_lines(alector_ref_path) for alector_ref_path in alector_ref_paths], lowercase=True)

# print(alector_sari)

# simplext_scores = get_all_scores(orig_sents=read_lines(simplext_complex_path), sys_sents=read_lines(simplext_ref_paths[0]), refs_sents=[read_lines(simplext_ref_path) for simplext_ref_path in simplext_ref_paths], lowercase=True)
# print("[simplext]:", simplext_scores)

# simplext_sari = corpus_sari(orig_sents=read_lines(simplext_complex_path), sys_sents=read_lines(simplext_ref_paths[0]), refs_sents=[read_lines(simplext_ref_path) for simplext_ref_path in simplext_ref_paths], lowercase=True)

# print(simplext_sari)

print("-"*10 + "Others" + "-"*10)
btts10 = read_lines(BTTS10_PATH)
asset_btts10_score = get_all_scores(orig_sents=read_lines(asset_complex_path), sys_sents=btts10, refs_sents=[read_lines(p) for p in _paths], lowercase=True)
print(asset_btts10_score)

muss_fr = read_lines(MUSS_fr_PATH)
muss_fr_fres = corpus_fkf_fr(muss_fr)
print(muss_fr_fres)

muss_fr_scores = get_all_scores(orig_sents=read_lines(alector_complex_path), sys_sents=muss_fr, refs_sents=[read_lines(alector_ref_path) for alector_ref_path in alector_ref_paths], lowercase=True)
print(muss_fr_scores)

muss_es = read_lines(MUSS_es_PATH)
muss_es_fres = corpus_fkf_es(muss_es)
print(muss_es_fres)

muss_es_scores = get_all_scores(orig_sents=read_lines(simplext_complex_path), sys_sents=muss_es, refs_sents=[read_lines(simplext_ref_path) for simplext_ref_path in simplext_ref_paths], lowercase=True)
print(muss_es_scores)

print(corpus_fkf_fr(read_lines('./sys_outputs/mBART_fr.txt')), corpus_fkf_es(read_lines('./sys_outputs/mBART_es.txt')))
print(corpus_fkf_fr(read_lines('./sys_outputs/transformer_fr.txt')), corpus_fkf_es(read_lines('./sys_outputs/transformer_es.txt'))),

print(corpus_fkf_fr(read_lines('./datasets/alector/alector.test.simple')), corpus_fkf_es(read_lines('./datasets/simplext/simplext.test.simple')))