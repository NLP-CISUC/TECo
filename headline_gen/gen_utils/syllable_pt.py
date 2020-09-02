class Syllables:
    vowels = ['a', 'e', 'i', 'o', 'u']
    consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'z', '\u00E7']
    group_1 = ['b', 'c', 'd', 'f', 'g', 'p', 't', 'v']
    group_2 = ['c', 'l', 'n']
    laterals = ['l', 'r', 'z']
    h = ['h']
    group_5 = ['c', 'g', 'm', 'p']
    group_6 = ['n']

    accents_ga = ['\u00E0', '\u00E1', '\u00E9', '\u00ED', '\u00F3', '\u00FA']
    circumflex = ['\u00E2', '\u00EA', '\u00EE', '\u00F4']
    tilde = ['\u00E3', '\u00F5']
    accents = ['\u00E0', '\u00E1', '\u00E9', '\u00ED', '\u00F3', '\u00FA',
               '\u00E2', '\u00EA', '\u00EE', '\u00F4', '\u00E3', '\u00F5']
    vowels_accents = ['a', 'e', 'i', 'o', 'u', '\u00E0', '\u00E1', '\u00E9', '\u00ED', '\u00F3',
                      '\u00FA', '\u00E2', '\u00EA', '\u00EE', '\u00F4', '\u00E3', '\u00F5']
    semi_vowels = ['i', 'u']
    semi_vowels_accents = ['i', 'u', '\u00E0', '\u00E1', '\u00E9', '\u00ED', '\u00F3',
                           '\u00FA', '\u00E2', '\u00EA', '\u00EE', '\u00F4', '\u00E3', '\u00F5']
    nasal = ['m', 'n']
    letters_q_g = ['q', 'g']
    hyphen = ['-']

    atonos = ["o", "a", "os", "as", "um", "uns", "me", "te", "se", "lhe", "nos", "lhes",
              "que", "com", "de", "por", "sem", "sob", "e", "mas", "nem", "ou"]
    regex_exception = "[qg]u[aei]"

    def __init__(self, word):
        self.word = word
        self.positions = []

    # divides a word into syllables
    def make_division(self):
        self.word = self.word.lower()

        if self.word == "ao" or self.word == "aos":
            return [self.word]

        self.build_positions()
        return self.fill_syllables()

    def build_positions(self):
        # first passage: separate vowels that have a consonant prior to them
        for i in range(1, len(self.word)):
            if self.word[i] in self.vowels_accents and self.word[i-1] not in self.vowels_accents and i > 1:
                if (self.word[i-1] in self.h and self.word[i-2] in self.group_2) or (self.word[i-1] in self.laterals and self.word[i-2] in self.group_1):
                    self.positions.append(i-2)
                else:
                    self.positions.append(i-1)

        # if a single consonant is alone in the first syllable
        if len(self.positions) > 0 and self.positions[0] == 1 and self.word[0] not in self.vowels_accents:
            self.positions[0] = 0
            # print("consoante sozinha na primeira silaba")

        # when it starts with two consonants, add a zero in the beginning
        if len(self.positions) == 0 or self.positions[0] != 0:
            self.positions.insert(0, 0)
            # print("duas consoantes no inicio")

        # second passage: separate ascending diphthongs
        for i in range(1, len(self.word)):
            if self.word[i] in self.vowels_accents and self.word[i-1] in self.vowels_accents:
                # if it has "qu" or "gu" or if the letter is an 'i' or 'u', doesn't separate
                if (i > 1 and self.word[i-1] == 'u' and self.word[i-2] in self.letters_q_g) or self.word[i-1] in self.tilde:
                    # print(self.word[i])
                    continue
                elif self.word[i] in self.semi_vowels:
                    # check for last syllable
                    if (i+1 < len(self.positions) and self.word[i+1] in self.laterals) and (self.word[i+1] not in self.nasal or (i+2 < len(self.positions) and self.word[i+2] in self.vowels_accents)):
                        # print(self.word[i])
                        continue

                for j in range(0, len(self.positions)):
                    if self.positions[j] > i:
                        self.positions.insert(j, i)
                        break
                    elif j == len(self.positions)-1:
                        self.positions.append(j)
                        break

    def fill_syllables(self):
        ret = []
        if len(self.positions) > 0:
            i = 0
            while i < len(self.positions)-1:
                ret.append(self.word[self.positions[i]:self.positions[i+1]])
                i += 1
            ret.append(self.word[self.positions[i]:])

        return ret
