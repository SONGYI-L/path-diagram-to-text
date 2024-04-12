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
Fine-tuning BART model and generating the explanatory text from the path diagram([bart](./BART%20fine-tune/bart%20fine-tuning.ipynb)).  
The sem-only fine-tuned BART model is [BART-sem](https://drive.google.com/drive/folders/1CCTphg1q12PZrqw1sbSdP45resW1eO7D?usp=drive_link)  
The webnlg-only fine-tuned BART model is [BART-webnlg](https://drive.google.com/drive/folders/1eWQQB22gsAYUjJiosqO9JTfNaFHXf8WY?usp=drive_link)  
The sem and webnlg-both fine-tuned BART model is [BART-sem-webnlg](https://drive.google.com/drive/folders/18XzlxcZELF82bfaJbi_mUnNv6HwDdRXV?usp=drive_link)  

Fine-tuning Vicuna v1.1 and Vicuna v1.5 models and generating the explanatory text from the path diagram([vicuna](./Vicuna%20fine-tune/vicuna_finetune_generate.ipynb)).  
The sem-only fine-tuned Vicuna v1.1 model is [vicuna-v1.1-sem](https://drive.google.com/drive/folders/1--qxAbNLNBZCz6mfU7ufqKYVb1H8zppD?usp=drive_link)  
The webnlg-only fine-tuned Vicuna v1.1 model is [vicuna-v1.1-webnlg](https://drive.google.com/drive/folders/1QDriTnGETuUhuaUPWXDq3tAfarrrBH3x?usp=drive_link)  
The sem and webnlg-both fine-tuned Vicuna v1.1 model is [vicuna-v1.1-sem-webnlg](https://drive.google.com/drive/folders/1AMc_WGhNUcS1j8wVCChwcxSLP_3porcu?usp=drive_link)  
The sem-only fine-tuned Vicuna v1.5 model is [vicuna-v1.5-sem](https://drive.google.com/drive/folders/1-1Hbd5O4Rn0PjpCaETiyt7NivS6rRgmZ?usp=drive_link)  
The webnlg-only fine-tuned Vicuna v1.5 model is [vicuna-v1.5-webnlg](https://drive.google.com/drive/folders/1NSpmPChyAwJGnk0zYr9LtjNVO3XqJ7Ab?usp=drive_link)  
The sem and webnlg-both fine-tuned Vicuna v1.5 model is [vicuna-v1.5-sem-webnlg](https://drive.google.com/drive/folders/1BBPpexE9mjMQeVIJ-fmKQMEo_UdZUbgl?usp=drive_link)  

###Evaluate  

