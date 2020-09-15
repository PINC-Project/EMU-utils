import argparse
import json
import sys
from pathlib import Path

import spacy
from spacy.parts_of_speech import PUNCT, SYM

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('annot', nargs='+', type=Path, help='path to EMU annotation file')
parser.add_argument('--model', default='en_core_web_md', help='annotation language')
parser.add_argument('--level', default='Word', help='name of annotation level containing orthographic transcription')
parser.add_argument('--label', default='Word', help='name of label within level containing orthographic transcription')
parser.add_argument('--ommit-syms', action='store_true', help='skip adding information about punctuation and symbols')
parser.add_argument('--backup', action='store_true', help='make a backup of the original')

args = parser.parse_args()

print('Loading spacy...')
nlp = spacy.load(args.model)
print('Done.')

for annot_file in tqdm(args.annot):
    with open(annot_file) as f:
        annotation = json.load(f)

    if args.backup:
        with open(str(annot_file) + '.bak', 'w') as f:
            json.dump(annotation, f, indent=4)

    level = None
    for l in annotation['levels']:
        if l['name'] == args.level:
            level = l
            break

    if not level:
        print(f'ERROR: level "{args.level}" not found in annotation')
        sys.exit(0)

    for item in level['items']:
        lab = []
        for l in item['labels']:
            if l['name'] not in ['lemma', 'POS', 'DEP']:
                lab.append(l)
        item['labels'] = lab

    label = args.label
    words = []
    ids = []
    for idx, item in enumerate(level['items']):
        for l in item['labels']:
            if l['name'] == label:
                word = l['value']
                if word and word[0] != '<':
                    words.append(word)
                    ids.append(idx)
                break

    idx_map = {}
    off = 0
    trans = ''
    for i, w in zip(ids, words):
        for n in range(len(w) + 2):
            idx_map[off + n] = i
        trans += w + ' '
        off += len(w) + 1

    doc = nlp(trans)
    word_annot = []
    for word in doc:
        if args.ommit_syms and word.pos == PUNCT or word.pos == SYM:
            continue
        word_annot.append((word.idx, word.orth_, word.lemma_, word.pos_, word.dep_))

    for word in word_annot:
        if word[0] not in idx_map:
            continue
        id = idx_map[word[0]]
        f_lem = True
        f_pos = True
        f_dep = True
        for l in level['items'][id]['labels']:
            if l['name'] == 'lemma':
                l['value'] += '+' + word[2]
                f_lem = False
            elif l['name'] == 'POS':
                l['value'] += '+' + word[3]
                f_pos = False
            elif l['name'] == 'DEP':
                l['value'] += '+' + word[4]
                f_dep = False
        if f_lem:
            level['items'][id]['labels'].append({'name': 'lemma', 'value': word[2]})
        if f_pos:
            level['items'][id]['labels'].append({'name': 'POS', 'value': word[3]})
        if f_dep:
            level['items'][id]['labels'].append({'name': 'DEP', 'value': word[4]})

    with open(annot_file, 'w') as f:
        json.dump(annotation, f, indent=4)
