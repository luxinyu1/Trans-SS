from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("./models/mbart-large-finetuned-opus-en-es-translation/")

model = AutoModelForSeq2SeqLM.from_pretrained("./models/mbart-large-finetuned-opus-en-es-translation/")