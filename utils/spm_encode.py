import sentencepiece as spm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model",
                    required=True,
                    default=None,
                    type=str,
                    help="The path to the spm model.")
parser.add_argument("--input", 
                    required=True,
                    help="The file to be encoded.")
parser.add_argument("--output", 
                    required=True,
                    help="The path to the output file.")

args = parser.parse_args()

s = spm.SentencePieceProcessor()
s.Load(args.model)

with open(args.input, "r", encoding="utf-8") as f_in:
    with open(args.output, "w+", encoding="utf-8") as f_out:
        count = 0
        for line in f_in:
            count += 1
            encoded_sentence = ' '.join(s.EncodeAsPieces(line.strip()))
            f_out.write(encoded_sentence + '\n')
            if count % 10000 == 0:
                print("processed {} lines".format(count))