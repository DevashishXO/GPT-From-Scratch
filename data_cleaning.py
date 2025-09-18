import pandas as pd
import re

df = pd.read_csv("dataset/premchand.tsv", sep="\t")
print(df.head())
print(df.info())

text_col = "Text" if "Text" in df.columns else df.columns[-1]  
raw_texts = df[text_col].dropna().astype(str).tolist()

def clean_text(t):
    # Removing Roman numerals or standalone numbers
    t = re.sub(r"\b\d+\b", " ", t)
    # Removing Devanagari digits (०१२३४५६७८९)
    t = re.sub(r"[०१२३४५६७८९]+", " ", t)
    # Removing weird symbols like » or decorative marks
    t = re.sub(r"[»“”‘’]", " ", t)

    # Adding a newline after every Hindi full stop (।)
    t = t.replace("।", "।\n")

    # Collapsing multiple spaces/newlines
    t = re.sub(r"[ \t]+", " ", t)  # collapse spaces
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)  # too many newlines → 2
    t = re.sub(r"\n\s+", "\n", t)  # clean spacing after newlines

    return t.strip()

# Cleaning each work
cleaned_texts = [clean_text(t) for t in raw_texts]

# Joining with double newlines between works
final_text = "\n\n".join(cleaned_texts)

MAX_CHARS = 1_200_000  # ~1.2M, slightly larger than Tiny Shakespeare
if len(final_text) > MAX_CHARS:
    final_text = final_text[:MAX_CHARS]

with open("dataset/input.txt", "w", encoding="utf-8") as f:
    f.write(final_text)

print(f"Total characters in cleaned text: {len(final_text)}")
