# path-diagram-to-text

## The structures and text in the path diagrams are first recognized, and then the path diagrams are converted into triple as input for the pre-trained models. Then the BART model, Vicuna v1.1 and Vicuna v1.7 models are fine-tuned and the explanatory text of the path diagram is generated.

### Recognize structures and text in path diagrams  
Using the Faster R-CNN（[fasterb](./Faster%20R-CNN/fasterb.ipynb)）to identify the different types of variables and relationships in the diagrams, using PP-OCR v3 model（[ocr](./OCR/ocr.ipynb)）to  identify the entity names in the diagrams and numerical values that indicate the strength of the relationships.

### Triple make
Convert path diagrams into triple based on the identified location information to be used as input for the pre-trained models.

### Text Generation with LLMs Model
