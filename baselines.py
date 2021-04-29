from access.utils.utils import get_data_filepath, get_dataset_dir, read_lines
from easse.report import get_all_scores
import nltk

def truncate(row_sentence, ratio, lang="english"):
    tokenized_sentence = nltk.word_tokenize(row_sentence, language=lang)
    target_words_in_sent = int(len(tokenized_sentence)*0.8)
    target_words = tokenized_sentence[:target_words_in_sent:]
    return " ".join(target_words)

turk_complex_path = get_data_filepath('turkcorpus', 'test', 'complex')
turk_ref_paths = [get_data_filepath('turkcorpus', 'test', 'simple.turk', i) for i in range(8)]

asset_complex_path = get_data_filepath('turkcorpus', 'test', 'complex') # same 2000 sentences
asset_ref_paths = [get_data_filepath('asset', 'test', 'simp', i) for i in range(10)]

newsela_complex_path = get_data_filepath('newsela', 'test', 'src')
newsela_ref_paths = [get_data_filepath('newsela', 'test', 'dst')]

print("-"*10 + "Identical Baseline" + "-"*10)

turk_scores = get_all_scores(orig_sents=read_lines(turk_complex_path), sys_sents=read_lines(turk_complex_path), refs_sents=[read_lines(turk_ref_path) for turk_ref_path in turk_ref_paths], lowercase=True)
print(turk_scores)

asset_scores = get_all_scores(orig_sents=read_lines(asset_complex_path), sys_sents=read_lines(asset_complex_path), refs_sents=[read_lines(asset_ref_path) for asset_ref_path in asset_ref_paths], lowercase=True)
print(asset_scores)

newsela_scores = get_all_scores(orig_sents=read_lines(newsela_complex_path), sys_sents=read_lines(newsela_complex_path), refs_sents=[read_lines(newsela_ref_path) for newsela_ref_path in newsela_ref_paths], lowercase=True)
print(newsela_scores)

print("-"*10 + "Truncate Baseline" + "-"*10)

turk_complex_sentences = read_lines(turk_complex_path)
turk_simple_sentences = [truncate(line, 0.8) for line in turk_complex_sentences]
turk_scores = get_all_scores(orig_sents=turk_complex_sentences, sys_sents=turk_simple_sentences, refs_sents=[read_lines(turk_ref_path) for turk_ref_path in turk_ref_paths], lowercase=True)
print(turk_scores)

asset_complex_sentences = read_lines(asset_complex_path)
asset_simple_sentences = [truncate(line, 0.8) for line in asset_complex_sentences]
asset_scores = get_all_scores(orig_sents=asset_complex_sentences, sys_sents=asset_simple_sentences, refs_sents=[read_lines(asset_ref_path) for asset_ref_path in asset_ref_paths], lowercase=True)
print(asset_scores)

newsela_complex_sentences = read_lines(newsela_complex_path)
newsela_simple_sentences = [truncate(line, 0.8) for line in newsela_complex_sentences]
newsela_scores = get_all_scores(orig_sents=newsela_complex_sentences, sys_sents=newsela_simple_sentences, refs_sents=[read_lines(newsela_ref_path) for newsela_ref_path in newsela_ref_paths], lowercase=True)
print(newsela_scores)

print("-"*10 + "Reference Baseline" + "-"*10)
