# This file is used to test the calculation of the metrics

from utils.calc import sentence_fkf, sentence_gfi, sentence_ari, sentence_dcrf, sentence_smog
from utils.paths import MODLES_DIR

# Test samples are collected from Wikipedia: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests

TOLERANCE = 1.0

appro_equal = lambda x,y: abs(x-y) < TOLERANCE

sent1 = 'The cat sat on the mat.'
sent2 = 'This sentence, taken as a reading passage unto itself, is being used to prove a point.'
sent3 = 'The Australian platypus is seemingly a hybrid of a mammal and reptilian creature.'

assert appro_equal(sentence_fkf(sent1), 116)
assert appro_equal(sentence_fkf(sent2), 69)
assert appro_equal(sentence_fkf(sent3), 37.5)

assert appro_equal(sentence_gfi(sent1), 2.4)
# assert appro_equal(sentence_gfi(sent2), 6.4)
# assert appro_equal(sentence_gfi(sent3), 17.51)

assert appro_equal(sentence_ari(sent1), -5.09)
assert appro_equal(sentence_ari(sent2), 6.59)
assert appro_equal(sentence_ari(sent3), 9.71)

assert appro_equal(sentence_dcrf(sent1), 0.30)
assert appro_equal(sentence_dcrf(sent2), 6.40)
assert appro_equal(sentence_dcrf(sent3), 12.78)

assert appro_equal(sentence_smog(sent1), 3)
assert appro_equal(sentence_smog(sent2), 3)
assert appro_equal(sentence_smog(sent3), 13.95)

print("Test complete, no calculation error!")
