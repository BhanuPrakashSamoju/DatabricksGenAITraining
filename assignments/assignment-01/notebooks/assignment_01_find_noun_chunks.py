# Databricks notebook source
# MAGIC %md
# MAGIC #Assignment for week 1
# MAGIC A simple python program to find all noun chunks in a sentence (with Spacy or NLTK)
# MAGIC  - Author: `Bhanu Prakash Samoju`

# COMMAND ----------

# Input sentence
sentence = "On a sunny afternoon, the children played in the park with their new toy. The tall trees swayed gently in the breeze while the dogs chased after the colorful frisbees. Nearby, a group of friends enjoyed a picnic, laughing and sharing stories under the big oak tree."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Spacy

# COMMAND ----------

!python -m spacy download en_core_web_sm

# COMMAND ----------

# MAGIC %md
# MAGIC Printing Noun Chunks in a sentence using SpaCy

# COMMAND ----------

import spacy

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Process the sentence
doc = nlp(sentence)

# Find and print all noun chunks
for chunk in doc.noun_chunks:
    print(chunk.text)


# COMMAND ----------

# MAGIC %md
# MAGIC Printing all Nouns in a sentence using SpaCy using Parts of Speech tag

# COMMAND ----------

# Find and print all nouns 
nouns_spacy = []
for token in doc:
    if token.tag_ in ["NN", "NNPS", "NNP", "NNS"]:
        nouns_spacy.append((token.text, token.pos_, token.tag_))
print(nouns_spacy)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using NLTK

# COMMAND ----------

# MAGIC %md
# MAGIC Printing Noun Chunks in a sentence using NLTK

# COMMAND ----------

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.chunk import RegexpParser
from nltk.tokenize import word_tokenize

# Tokenize and POS tag the sentence
tokens = word_tokenize(sentence)
tagged = pos_tag(tokens)

# Define a chunk grammar
grammar = r"""
NP: {<DT>?<JJ>*<NN>}
    {<NNP>+}  
    {<NNS>+}  
    {<NNPS>+}  
"""

# Create a chunk parser
cp = RegexpParser(grammar)

# Parse the sentence
tree = cp.parse(tagged)

# Extract and print noun phrases
for subtree in tree.subtrees():
    if subtree.label() == 'NP':
        print(" ".join(word for word, tag in subtree.leaves()))


# COMMAND ----------

tree.pretty_print()

# COMMAND ----------

# MAGIC %md
# MAGIC Printing Nouns in a sentence using NLTK

# COMMAND ----------

# Define a Noun grammar
noun_grammar = r"""
NOUN: {<NN>}    # Singular noun
      {<NNS>}   # Plural noun
      {<NNP>}   # Proper noun, singular
      {<NNPS>}  # Proper noun, plural
"""

# Create a chunk parser
cp = RegexpParser(noun_grammar)

# Parse the sentence
tree = cp.parse(tagged)

# Extract and print noun phrases
nouns = []
for subtree in tree.subtrees():
    if subtree.label() == 'NOUN':
        nouns.append(" ".join(word for word, tag in subtree.leaves()))
print("Nouns:", nouns)
