from bigrammer import Bigrammer
from argparse import ArgumentParser


if __name__ ==  '__main__':
    parser = ArgumentParser(description='Bigram analysis of a given text')
    parser.add_argument('text', help='path to .txt file for analysis')
    parser.add_argument('lemm_save', help='option to save lemmatized text (True or False)', type=bool)
    parser.add_argument('--lemm', help='path to lemm file to save lemmatized text')
    parser.add_argument('words', help='words separated by | (word1|word2|word3|...) for bigram frequency output')
    parser.add_argument('logging', help='option to make text log of bigram analysis results (True or False)', type=bool)
    parser.add_argument('--log', help='path to log file for analysis results')
    args = parser.parse_args()

    bigrammer = Bigrammer(args.text, make_lemm_text=args.lemm_save, lemm_path=args.lemm, make_log=args.logging)
    bigrammer.get_pos_lemm_frequencies('NOUN')
    bigrammer.get_pos_lemm_frequencies('INFN')
    bigrammer.get_pos_lemm_frequencies('ADJF')
    bigrammer.make_bigrams()
    bigrammer.bigram_measure_dice()
    bigrammer.bigram_measure_mi()
    bigrammer.dice_top_20()
    bigrammer.mi_top_20()
    for word in args.words.split('|'):
        bigrammer.bigram_frequency_by_word(word)
        bigrammer.dice_top_20_by_word(word)
        bigrammer.mi_top_20_by_word(word)
    if args.logging:
        bigrammer.write_log_to_path(args.log)
