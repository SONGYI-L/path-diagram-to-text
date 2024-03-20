# path-diagram-to-text

### The structures and text in the path diagrams are first recognized using the Faster R-CNN and PP-OCR v3 models, and then the path diagrams are converted into triple (triple make) based on the identified location information to be used as input for the pre-trained models. 

### Then the BART model, Vicuna v1.1 and Vicuna v1.7 models are fine-tuned and the explanatory text of the path diagram is generated.