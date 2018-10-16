import re
import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize, FreqDist, bigrams
from matplotlib import cm, colors
import numpy as np
from tqdm import tqdm
import pymorphy2


class Bigrammer:
    def __init__(self, text_file, make_lemm_text=False, lemm_path='', make_log=False):
        self.morph = pymorphy2.MorphAnalyzer()
        txt = open(text_file, "r")
        self.text = ''
        for line in tqdm(txt):
            self.text += line
        self.text = self.text.replace('`', '')
        self.text = self.text.replace('\'', '')
        self.text_symbols = re.sub(r'\s', '', self.text)
        self.text_letters = re.sub(r'[^А-Яа-я]', '', self.text)
        self.sentences = sent_tokenize(self.text)
        self.sent_tokens = [word_tokenize(s) for s in self.sentences]
        self.lemm_punct_list = []
        self.lemm_list = []
        for sent in tqdm(self.sent_tokens):
            sent_lemms = []
            sent_lemms_punct = []
            for token in sent:
                pos_tag = self.morph.parse(token)[0][1]
                if str(pos_tag) != 'PNCT':
                    sent_lemms.append(self.morph.parse(token)[0].normal_form)
                    sent_lemms_punct.append(self.morph.parse(token)[0].normal_form)
                else:
                    sent_lemms_punct.append(token)
            self.lemm_list.append(sent_lemms)
            self.lemm_punct_list.append(' '.join(sent_lemms_punct))
        self.lemm_text = ' '.join([' '.join(sent) for sent in self.lemm_list])
        if make_lemm_text:
            with open(lemm_path, 'w') as lemm_file:
                for sent in tqdm(self.lemm_punct_list):
                    lemm_file.write(sent + '\r\n')
        self.log_flag = False
        self.bigrams = []
        self.wordfreq = FreqDist(word_tokenize(self.lemm_text))
        self.bigram_dice = {}
        self.bigram_mi = {}
        if make_log:
            self.log = ''
            self.log_flag = True

    def get_pos_lemm_frequencies(self, pos):
        pos_list = []
        for sent in tqdm(self.lemm_list):
            for word in sent:
                pos_tag = self.morph.parse(word)[0].tag.POS
                if pos_tag == pos:
                    pos_list.append(word)
        fdist = FreqDist(pos_list)
        print('Леммы части речи %s, их частоты, их относительные частоты (топ-10):' % pos)
        if self.log_flag:
            self.log += 'Леммы части речи %s, их частоты, их относительные частоты (топ-10):' % pos + '\n'
        num = 1
        for word, frequency in fdist.most_common(10):
            print('%d. %s %d %.2f' % (num, word, frequency, frequency / len(pos_list)))
            if self.log_flag:
                self.log += '%d. %s %d %.2f' % (num, word, frequency, frequency / len(pos_list)) + '\n'
            num += 1
        print('----------------------------------------------------------\n')
        if self.log_flag:
            self.log += '----------------------------------------------------------\n'

    def make_bigrams(self):
        for sent in self.lemm_list:
            bigrm = bigrams(sent)
            for word1, word2 in bigrm:
                self.bigrams.append(word1 + ' ' + word2)

    def word_frequency(self, word):
        return self.wordfreq[word]

    def bigram_frequency_by_word(self, word):
        bidist = dict(FreqDist(self.bigrams))
        counter = 0
        print('Самые частотные биграммы со словом %s' % word)
        if self.log_flag:
            self.log += 'Самые частотные биграммы со словом %s' % word
        for bigram in sorted(bidist.items(), key=lambda kv: kv[1], reverse=True):
            if word in bigram[0]:
                if counter < 20:
                    print(bigram)
                    if self.log_flag:
                        self.log += bigram
                    counter += 1
                else:
                    break
        print('----------------------------------------------------------\n')
        if self.log_flag:
            self.log += '----------------------------------------------------------\n'

    def bigram_measure_dice(self):
        for bigram, value in dict(FreqDist(self.bigrams)).items():
            word1 = bigram.split()[0]
            word2 = bigram.split()[1]
            word1_freq = self.word_frequency(word1)
            word2_freq = self.word_frequency(word2)
            try:
                dice = 2 * value / (word1_freq + word2_freq)
            except:
                dice = 0
            self.bigram_dice[bigram] = dice

    def bigram_measure_mi(self):
        corpus_size = len(np.unique(self.lemm_list))
        for bigram, value in dict(FreqDist(self.bigrams)).items():
            word1 = bigram.split()[0]
            word2 = bigram.split()[1]
            word1_freq = self.word_frequency(word1)
            word2_freq = self.word_frequency(word2)
            try:
                mi = np.log2(corpus_size * value / (word1_freq * word2_freq))
            except:
                mi = 0
            self.bigram_mi[bigram] = mi

    def dice_top_20(self):
        counter = 0
        print('Самые устойчивые биграммы согласно мере Dice')
        if self.log_flag:
            self.log += 'Самые устойчивые биграммы согласно мере Dice'
        for bigram in sorted(self.bigram_dice.items(), key=lambda kv: kv[1], reverse=True):
            if counter < 20:
                print(bigram)
                if self.log_flag:
                    self.log += bigram
                counter += 1
            else:
                break
        print('----------------------------------------------------------\n')
        if self.log_flag:
            self.log += '----------------------------------------------------------\n'

    def mi_top_20(self):
        counter = 0
        print('Самые устойчивые биграммы согласно мере MI')
        if self.log_flag:
            self.log += 'Самые устойчивые биграммы согласно мере MI'
        for bigram in sorted(self.bigram_mi.items(), key=lambda kv: kv[1], reverse=True):
            if counter < 20:
                print(bigram)
                if self.log_flag:
                    self.log += bigram
                counter += 1
            else:
                break
        print('----------------------------------------------------------\n')
        if self.log_flag:
            self.log += '----------------------------------------------------------\n'

    def dice_top_20_by_word(self, word):
        counter = 0
        print('Самые устойчивые биграммы согласно мере Dice со словом %s' % word)
        if self.log_flag:
            self.log += 'Самые устойчивые биграммы согласно мере Dice со словом %s' % word
        for bigram in sorted(self.bigram_dice.items(), key=lambda kv: kv[1], reverse=True):
            if word + ' ' in bigram[0]:
                if counter < 20:
                    print(bigram)
                    if self.log_flag:
                        self.log += bigram
                    counter += 1
                else:
                    break
        print('----------------------------------------------------------\n')
        if self.log_flag:
            self.log += '----------------------------------------------------------\n'

    def mi_top_20_by_word(self, word):
        counter = 0
        print('Самые устойчивые биграммы согласно мере MI со словом %s' % word)
        if self.log_flag:
            self.log += 'Самые устойчивые биграммы согласно мере MI со словом %s' % word
        for bigram in sorted(self.bigram_mi.items(), key=lambda kv: kv[1], reverse=True):
            if word + ' ' in bigram[0]:
                if counter < 20:
                    print(bigram)
                    if self.log_flag:
                        self.log += bigram
                    counter += 1
                else:
                    break
        print('----------------------------------------------------------\n')
        if self.log_flag:
            self.log += '----------------------------------------------------------\n'

    def write_log_to_path(self, log_path):
        with open(log_path, "w") as log_text:
            log_text.write(self.log)
