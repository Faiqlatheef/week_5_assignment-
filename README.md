
# Week 5 Assignment: Named Entity Recognition (NER) with DistilBERT

## Overview  
This repository contains the solution to the **Named Entity Extraction** task for Week 5, using the **WikiAnn NER dataset** with the **DistilBERT model**. The objective is to build a robust NER model, train it, and evaluate its performance using relevant metrics. The code is implemented in **PyTorch** using the HuggingFace `transformers` and `datasets` libraries.

---

## Dataset  
We use the **WikiAnn** dataset (English language), which contains annotated text with named entities such as locations, persons, and organizations.

Dataset source: [WikiAnn on HuggingFace](https://huggingface.co/datasets/wikiann)

---

## Model Architecture  
The model uses **DistilBERT**, a lightweight transformer model for better efficiency. The output from DistilBERT is passed through a linear layer to predict the NER tags.  
- **Base Model:** `distilbert-base-cased`  
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** AdamW with linear learning rate scheduler  
- **Training Strategy:** Early stopping and learning rate scheduling for better performance.

---

## Installation  

### Prerequisites  
Make sure you have Python 3.7+ installed. Then, install the required dependencies:

```bash
pip install torch transformers datasets scikit-learn
```

---

## How to Run  

1. **Clone the repository:**

```bash
git clone https://github.com/your_username/week_5_assignment.git
cd week_5_assignment
```

2. **Train the model:**  
Run the `train.py` script to train the model:

```bash
python train.py
```

3. **Evaluate the model:**  
After training, you can evaluate the model using the `evaluate.py` script:

```bash
python evaluate.py
```

---

## Directory Structure  

```
week_5_assignment/
│
├── train.py           # Script to train the NER model
├── evaluate.py        # Script to evaluate the trained model
├── model.py           # Contains the NER model class
├── requirements.txt   # Dependencies file
├── README.md          # This file
└── data/              # Directory for processed data (optional)
```

---

## Results  
After training, the following metrics were achieved on the validation set:

| Metric        | Score   |
|---------------|---------|
| Precision     | 85.2%   |
| Recall        | 83.7%   |
| F1-Score      | 84.4%   |

---

## Bonus Tasks Implemented  
- **Early Stopping:** The training stops if the validation loss does not improve for 3 consecutive epochs.  
- **Learning Rate Scheduling:** A linear learning rate scheduler is applied with warm-up steps.

---

## References  
1. [HuggingFace Transformers](https://huggingface.co/transformers/)  
2. [WikiAnn Dataset](https://huggingface.co/datasets/wikiann)  
3. [PyTorch](https://pytorch.org/)

---

## License  
This project is licensed under the MIT License.

---

## Author  
- **Faiq latheef** – [GitHub Profile](https://github.com/Faiqlatheef)

---

## Acknowledgments  
Special thanks to the instructors and teaching assistants for their guidance on this assignment.
