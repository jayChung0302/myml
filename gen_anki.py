# this file is for utilizing genanki source code. see genanki library. 
# https://github.com/kerrickstaley/genanki

import random
import genanki
from typing import List, Dict
from tools import pkl_dump, pkl_load

def add_anki(deck_name : str, q_and_a : Dict, deck_dict: Dict=None):
    # param:q_and_a: {'str1':'str1', 'str2':'str2', ...}
    # return: updated deck_dict
    # generate "output.apkg" data
    if not deck_dict:
        deck_dict = pkl.load('./anki_dict.pkl')

    if deck_name not in deck_dict:
        seed = random.randrange(1<<30, 1<<31)
        deck_dict[deck_name] = seed
    else:
        seed = deck_dict[deck_name]
    
    my_model = genanki.Model(
        seed,
        'Simple Model',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'},
        ],
        templates=[
            {
            'name': 'Card 1',
            'qfmt': '{{Question}}',
            'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            },
        ])

    my_deck = genanki.Deck(
            seed,
            deck_name)

    for key in q_and_a.keys():
        my_note = genanki.Note(
            model=my_model,
            fields=[key, q_and_a[key]])
        
        my_deck.add_note(my_note)

    genanki.Package(my_deck).write_to_file('./output.apkg')

    pkl_dump(deck_dict, name='anki_save_data')
    # return deck_dict


if __name__ == '__main__':
    deck_dict = {'ex-auto-anki': random.randrange(1 << 30, 1 << 31)}
    q_and_a = {
        '1':'2',
        'auto':'mate',
        'input':'output'
    }
    add_anki('ex-auto-anki', q_and_a, deck_dict)
