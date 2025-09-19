# GPT From Scratch

This project is an implementation of a Generative Pre-trained Transformer (GPT) model built entirely from scratch, inspired by [Andrej Karpathy's "Let's Build GPT from Scratch" tutorial](https://youtu.be/kCc8FmEb1nY).

## Objective

The goal is to deeply understand the inner workings of GPT models by building one step-by-step, without relying on high-level machine learning frameworks.

## Dataset
I have used the 'Premchand Corpus' dataset (downsized from 6.7M characters to 1.2M characters to match the character count in Karpathy's tiny-shakespeare dataset) available on Kaggle for training the transformer model because it contains a rich collection of literary works by the renowned Indian author Munshi Premchand. This dataset provides a diverse range of text, making it suitable for training a language model like GPT. You can find the dataset [here](https://www.kaggle.com/datasets/amankhandelia/premchand-corpus/data).

## Notes:
- I first truncated the dataset to 1.2M characters but then had to manually adjust the dataset because the last poem was cut off abruptly which is not ideal for training. The current character count is 1219833 which is still close to 1.2M.


## Credits

Special thanks to [Andrej Karpathy](https://github.com/karpathy) for his excellent tutorial and educational resources, which serve as the foundation for this project.

---
