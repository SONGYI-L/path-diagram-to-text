# Path Model Textualization of SEM Results based on Deep Learning

## The structures and text in the path diagrams are first recognized, and then the path diagrams are converted into triple as input for the pre-trained models. Then generate text with LLMs models.

### Recognize structures and text in path diagrams  
Using the Faster R-CNN（[fasterb](./Faster%20R-CNN/fasterb.ipynb)）to identify the different types of variables and relationships in the diagrams.   
The VOCdevkit data used in fasterb is addressed to：[VOCdevkit](https://drive.google.com/drive/folders/1p83AQXnND1E0L-8fe6wyJ5h3iYOmedtS?usp=sharing)  
The address of the model_path used for training in fasterb is: [model path](https://drive.google.com/file/d/1p4p4ggyRxQf6Lj71_lWvQR7x2oP27VWU/view?usp=drive_link)  
To make predictions using own trained model you have to modify the model_path in [frcnn.py](./Faster%20R-CNN/frcnn.py)  
Using PP-OCR v3 model（[ocr](./OCR/ocr.ipynb)）to  identify the entity names in the diagrams and numerical values that indicate the strength of the relationships.  
The data used in OCR is addressed to：[OCR](https://drive.google.com/drive/folders/1zYbr7nK6TnTxJvXbI4u5pEhUzA6jZeds?usp=drive_link)

### Triple make
Convert path diagrams into triple based on the identified location information to be used as input for the pre-trained models.  
The data used in triple-make is addressed to：[triple make data](https://drive.google.com/drive/folders/11_IeStniuELiaVb5CMHKkATcUzlikTAq?usp=drive_link)

### Text Generation with LLMs Model  
#### BART model  
Fine-tuning BART model and generating the explanatory text from the path diagram([bart](./BART%20fine-tune/bart%20fine-tuning.ipynb)).  
For BART model the data of fine-tuning and test are:([sem data for bart](./BART%20fine-tune/sem%20data)) and ([webnlg data for bart](./BART%20fine-tune/webnlg%20data)).  

The sem-only fine-tuned BART model is [BART-sem](https://drive.google.com/drive/folders/1CCTphg1q12PZrqw1sbSdP45resW1eO7D?usp=drive_link)  
The webnlg-only fine-tuned BART model is [BART-webnlg](https://drive.google.com/drive/folders/1eWQQB22gsAYUjJiosqO9JTfNaFHXf8WY?usp=drive_link)  
The sem and webnlg-both fine-tuned BART model is [BART-sem-webnlg](https://drive.google.com/drive/folders/18XzlxcZELF82bfaJbi_mUnNv6HwDdRXV?usp=drive_link)  

#### Vicuna models
Fine-tuning Vicuna v1.1 and Vicuna v1.5 models and generating the explanatory text from the path diagram([vicuna](./Vicuna%20fine-tune/vicuna_finetune_generate.ipynb)).  
For Vicuna models the data of fine-tuning are:([sem](https://huggingface.co/datasets/LLLsy/sem))、([webnlg](https://huggingface.co/datasets/LLLsy/webnlg))、([sem and webnlg](https://huggingface.co/datasets/LLLsy/sem_webnlg))  
For Vicuna models the data of test is:([sem data for vicuna](./Vicuna%20fine-tune/sem%20data%20for%20vicuna))  

The sem-only fine-tuned Vicuna v1.1 model is [vicuna-v1.1-sem](https://drive.google.com/drive/folders/1--qxAbNLNBZCz6mfU7ufqKYVb1H8zppD?usp=drive_link)  
The webnlg-only fine-tuned Vicuna v1.1 model is [vicuna-v1.1-webnlg](https://drive.google.com/drive/folders/1QDriTnGETuUhuaUPWXDq3tAfarrrBH3x?usp=drive_link)  
The sem and webnlg-both fine-tuned Vicuna v1.1 model is [vicuna-v1.1-sem-webnlg](https://drive.google.com/drive/folders/1AMc_WGhNUcS1j8wVCChwcxSLP_3porcu?usp=drive_link)  
The sem-only fine-tuned Vicuna v1.5 model is [vicuna-v1.5-sem](https://drive.google.com/drive/folders/1-1Hbd5O4Rn0PjpCaETiyt7NivS6rRgmZ?usp=drive_link)  
The webnlg-only fine-tuned Vicuna v1.5 model is [vicuna-v1.5-webnlg](https://drive.google.com/drive/folders/1NSpmPChyAwJGnk0zYr9LtjNVO3XqJ7Ab?usp=drive_link)  
The sem and webnlg-both fine-tuned Vicuna v1.5 model is [vicuna-v1.5-sem-webnlg](https://drive.google.com/drive/folders/1BBPpexE9mjMQeVIJ-fmKQMEo_UdZUbgl?usp=drive_link)  

### Evaluate  
Using the BLUE, METEOR, and ROUGE score to evaluate the explanatory text:
```
python /fastchat.py '/target.target' '/output.json'
```

Using the ROUGE-1, ROUGE-2, ROUGE-S score to evaluate the explanatory text:
```Python
import json
from rouge import Rouge
from itertools import combinations


def read_target_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 将文本块分隔开，每个块代表一个项
    blocks = '\n'.join(lines).split('\n\n')
    # 去除每个块中的空行，并连接成单一字符串
    texts = [' '.join(block.split('\n')) for block in blocks if block.strip() != '']
    return texts

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return [item['text'] if item['text'].strip() not in {"", ".", ","} else "PLACEHOLDER" for item in data]

def calculate_rouge_scores(references, hypotheses):
    # 确保references和hypotheses的长度一致
    print(len(references))
    print(len(hypotheses))
    assert len(references) == len(hypotheses), "References and Hypotheses lengths do not match."

    # 检查是否所有的假设文本都是 "PLACEHOLDER"
    all_placeholders = all(text == "PLACEHOLDER" for text in output_texts)
    print("所有假设文本都是'PLACEHOLDER'？", all_placeholders)

    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)

    return scores

# 生成跳跃bigrams的生成器，而不是集合
def create_skip_bigrams(text, skip=4):
    words = text.split()
    for i in range(len(words)):
        for j in range(1, skip+1):
            if i + j < len(words):
                yield (words[i], words[i + j])

# 计算ROUGE-S分数
def calculate_rouge_s(reference, hypothesis, skip=4):
    ref_skip_bigrams = set(create_skip_bigrams(reference, skip))
    hyp_skip_bigrams = set(create_skip_bigrams(hypothesis, skip))

    if not ref_skip_bigrams or not hyp_skip_bigrams:
        return 0.0

    overlap = ref_skip_bigrams.intersection(hyp_skip_bigrams)
    recall = len(overlap) / len(ref_skip_bigrams)
    precision = len(overlap) / len(hyp_skip_bigrams)
    if recall + precision != 0:
        f1 = 2 * (recall * precision) / (recall + precision)
    else:
        f1 = 0.0
    return f1

# 读取文件
target_texts = read_target_file('/content/drive/MyDrive/JointGT-main/data/webnlg/test.target')
output_texts = read_json_file('/content/drive/MyDrive/JointGT-main/bart/sem_fintune/bart_fintune_output.json')

# 计算ROUGE-1和ROUGE-2分数
rouge_scores = calculate_rouge_scores(target_texts, output_texts)
print("ROUGE-1:", rouge_scores['rouge-1'])
print("ROUGE-2:", rouge_scores['rouge-2'])

# 计算每对文本的ROUGE-S分数，并计算平均分数
rouge_s_scores = [calculate_rouge_s(ref, hyp) for ref, hyp in zip(target_texts, output_texts)]
average_rouge_s = sum(rouge_s_scores) / len(rouge_s_scores)
print("ROUGE-S Average:", average_rouge_s)
```
