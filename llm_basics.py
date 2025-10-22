#### Practicing LLM basics --- using Hugging Face pretrained models

## packages

from transformers import pipeline

## summarize

## summarization based on the news data

summarizer = pipeline(task= "summarization",
                          model = "facebook/bart-large-cnn")

crevecoeur = "What then is the American, this new man? He is either an European, or the descendant of an European, hence that strange mixture of blood, which you will find in no other country. I could point out to you a family whose grandfather was an Englishman, whose wife was Dutch, whose son married a French woman, and whose present four sons have now four wives of different nations. He is an American, who leaving behind him all his ancient prejudices and manners, receives new ones from the new mode of life he has embraced, the new government he obeys, and the new rank he holds. He becomes an American by being received in the broad lap of our great Alma Mater. Here individuals of all nations are melted into a new race of men, whose labours and posterity will one day cause great changes in the world. Americans are the western pilgrims, who are carrying along with them that great mass of arts, sciences, vigour, and industry which began long since in the east; they will finish the great circle. The Americans were once scattered all over Europe; here they are incorporated into one of the finest systems of population which has ever appeared, and which will hereafter become distinct by the power of the different climates they inhabit. The American ought therefore to love this country much better than that wherein either he or his forefathers were born. Here the rewards of his industry follow with equal steps the progress of his labour; his labour is founded on the basis of nature, self-interest; can it want a stronger allurement? Wives and children, who before in vain demanded of him a morsel of bread, now, fat and frolicsome, gladly help their father to clear those fields whence exuberant crops are to arise to feed and to clothe them all; without any part being claimed, either by a despotic prince, a rich abbot, or a mighty lord. I lord religion demands but little of him; a small voluntary salary to the minister, and gratitude to God; can he refuse these? The American is a new man, who acts upon new principles; he must therefore entertain new ideas, and form new opinions. From involuntary idleness, servile dependence, penury, and useless labour, he has passed to toils of a very different nature, rewarded by ample subsistence. --This is an American. "


summary = summarizer(crevecoeur, min_length = 20, max_length = 100)

print(summary[0]["summary_text"])


####### Translation


# Korean to English

translator_k2e = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="kor_Hang",
    tgt_lang="eng_Latn",
    device_map="auto"
)

print(translator_k2e("좋은 아침이에요! 오늘 어때요?")[0]["translation_text"])
# something like "Good morning, how are you today?"

# English to Korean

translator_e2k = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    tgt_lang="kor_Hang",
    src_lang = "eng_Latn",
    device_map="auto"
)

print(translator_e2k("My hovercraft is full of eels")[0]["translation_text"])

# From Monty Python's dirty Hungarian phrasebook, but this seems to give "frog" instead of eel 


#### German to English
## More precise DE-EN model

translator_e2g = pipeline("translation", model="facebook/wmt19-en-de")

text2 = "Good morning! How are you today?"
translation = translator_e2g(text2)
print(translation[0]['translation_text'])

## This gives the formal "you" ("Ihnen")


text3 = "My hovercraft is full of eels."
translation = translator_e2g(text3)
print(translation[0]['translation_text'])

## "Luftkissenboot" for Hovercraft and "Aale" for eels

## More general language model

translator_e2g2 = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang = "eng_Latn",
    tgt_lang="deu_Latn",
    device_map="auto"
)

print(translator_e2g2("My hovercraft is full of eels")[0]["translation_text"])

## This says "Flugzeug" which is airplane instead of hovercraft, and "Eingeweide" for Eels

