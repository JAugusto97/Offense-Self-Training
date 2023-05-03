# %%
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import pandas as pd
from utils import normalize_tweet

# %%
df = pd.read_csv("datasets/tweets.csv")
df = df[["tweet_id", "text"]]
df["text"] = df["text"].apply(normalize_tweet)
text = df["text"].tolist()

# %%
back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de', 
    to_model_name='facebook/wmt19-de-en'
)
back_translated_text = back_translation_aug.augment(text)
df["backtranslation"] = back_translated_text

# %%
synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.3, aug_min=1, aug_max=None)
synonym_augmented_text = synonym_aug.augment(text)
df["synonym_substitution"] = synonym_augmented_text

# %%
randomswap_aug = naw.RandomWordAug(action="swap")
randomswap_augmented_text = randomswap_aug.augment(text)
df["word_swap"] = randomswap_augmented_text

# %%
df.to_csv("datasets/tweets_augmented.csv", index=False)


