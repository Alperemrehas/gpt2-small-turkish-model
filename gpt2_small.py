from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
from transformers import RobertaTokenizer
import torch
import os

print("Started")
os.environ['CURL_CA_BUNDLE'] = ''

PATH= "C:/Users/89950000/Documents/gpt2-small-turkish/"
#model = torch.load('./pytorch_model.bin',map_location=torch.device('cpu'))


#model = AutoModelWithLMHead.from_pretrained(PATH, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(PATH)

#tokenizer = RobertaTokenizer.from_pretrained(PATH, local_files_only=True)
print("Step1")
model = AutoModelForCausalLM.from_pretrained(PATH, local_files_only=True)
print("Done")
# Get sequence length max of 1024
tokenizer.model_max_length=1024 

#model.load_state_dict(state_dict)
#model.eval()  # disable dropout (or leave in train mode to finetune)

#Generate 1 word
# input sequence
text = "Bu yazıyı bilgisayar yazdı."
inputs = tokenizer(text, return_tensors="pt")

# model output
outputs = model(**inputs, labels=inputs["input_ids"])
loss, logits = outputs[:2]
predicted_index = torch.argmax(logits[0, -1, :]).item()
predicted_text = tokenizer.decode([predicted_index])

# results
print('input text:', text)
print('predicted text:', predicted_text)

# input text: 
# predicted text:  

#Generate Full Sequence
# input sequence
text = input("Size Nasıl Yardımcı Olabilirim?: ")
inputs = tokenizer(text, return_tensors="pt")

# model output using Top-k sampling text generation methodN
sample_outputs = model.generate(inputs.input_ids,
                                pad_token_id=50256,
                                do_sample=True, 
                                max_length=500, # put the token number you want
                                top_k=40,
                                num_return_sequences=1)

# generated sequence
for i, sample_output in enumerate(sample_outputs):
    print(">> Generated text {} {} : ".format(i+1, tokenizer.decode(sample_output.tolist())))