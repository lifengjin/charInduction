import bidict

# phones = {'p0', 'ph', 'pp', 't0', 'th', 'tt', 'k0', 'kh', 'kk', 's0', 'ss', 'h0', 'c0', 'ch', 'cc',
#           'mm', 'nn', 'rr', 'pf', 'ph', 'tf', 'th', 'kf', 'kh', 'kk', 's0', 'ss', 'h0', 'c0', 'ch',
#           'mf', 'nf', 'ng', 'll', 'ks', 'nc', 'nh', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh', 'ps',
#           'ii', 'ee', 'qq', 'aa', 'xx', 'vv', 'uu', 'oo', 'ye', 'yq', 'ya', 'yv', 'yu', 'yo', 'wi',
#           'wo', 'wq', 'we', 'wa', 'wv', 'xi'} # 66 of them, but some repeats
def get_korean_phone_mappings():
    phones = ['aa', 'ch', 'ss', 'cc', 'h0', 'ii', 'xx', 'xi', 'lh', 'vv', 'wo', 'nh', 'ps', 'pp', 'lb', 'wa', 'rr', 'tt', 's0',
              'lk', 'nf', 'ng', 'ee', 'lt', 'k0', 'pf', 'ls', 'th', 'tf', 'ph', 'yv', 'oo', 'we', 'lp', 'yo', 'ya', 'ye', 'nc',
              'wi', 'wv', 'wq', 'yq', 'c0', 'yu', 'kf', 'qq', 'lm', 'p0', 'mf', 'nn', 'mm', 'kh', 't0', 'ks', 'kk', 'uu', 'll'] # no repeats, 57

    begin = int('4e00', 16)

    phone_mappings = bidict.bidict()

    for phone in phones:
        identifier = chr(begin)
        phone_mappings[phone] = identifier
        begin += 1

    return phone_mappings


def translate_phone_to_ids(string, mappings):
    translated_string = ''
    for char_index in range(0, len(string), 2):
        bichars = string[char_index:char_index+2]
        if bichars in mappings:
            translated_string += mappings[bichars]
        else:
            translated_string += bichars
    return translated_string

def translate_ids_to_phones(string, mappings):
    translated_string = ''
    for char in string:
        if char in mappings.inv:
            translated_string += mappings.inv[char]
        else:
            translated_string += char
    return translated_string