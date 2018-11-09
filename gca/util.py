# coding=utf-8

import re

english_word_regex = re.compile("[a-zA-Z]+")


def is_english_word(word):
    return english_word_regex.fullmatch(word)