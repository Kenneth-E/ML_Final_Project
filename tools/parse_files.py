import json
import random

file_name_and_language_pairs = [
    ["Bible_texts/ACV.txt", "English"],
    ["Bible_texts/CebPinadayag.txt", "Cebuano"],
    ["Bible_texts/CzeCSP.txt", "Czech"],
    ["Bible_texts/DutSVV.txt", "Dutch"],
    ["Bible_texts/Esperanto.txt", "Spanish"],
    ["Bible_texts/FinBiblia.txt", "Finnish"],
    ["Bible_texts/FrePGR.txt", "French"],
    ["Bible_texts/Haitian.txt", "Haitian"],
    ["Bible_texts/LvGluck8.txt", "Latvian"],
    ["Bible_texts/NorSMB.txt", "Norwegian"],
    ["Bible_texts/PolGdanska.txt", "Polish"],
    ["Bible_texts/PorNVA.txt", "Portugese"],
    ["Bible_texts/Swe1917.txt", "Swedish"]
]
languange_sample_pair_list = []

for pair in file_name_and_language_pairs:
    with open(pair[0], 'r', encoding='utf-8') as f:
        language = pair[1]
        lines = f.readlines() # list containing lines of file

        i = 1
        for line in lines:
            line = line.strip() # remove leading/trailing white spaces
            if (len(line) > 50) and (len(line) < 500):
                languange_sample_pair_list.append([language, line]) # append line

# shuffle list
random.shuffle(languange_sample_pair_list)

# pretty printing lines
print(json.dumps(languange_sample_pair_list[0:100], indent=4))