import pandas as pd
import matplotlib.pyplot as plt
# #--- forms.txt --------------------------------------------------------------#
#
# iam database form information
#
# format: a01-000u 000 2 prt 7 5 52 36
#
#     a01-000u  -> form id
#     000       -> writer id
#     2         -> number of sentences
#     prt       -> word segmentation
#                     prt: some lines correctly segmented
#                     all: all lines correctly segmented
#     7 5       -> 5 of 7 lines are correctly segmented into words
#     52 36     -> the form contains 52 words,
#                  36 are in lines which have been correctly segmented
forms_files_str = ['A-D', 'E-H', 'I-Z']

import re
def read_forms_dataframe(set_image_path=True):
    with open('data/forms.txt', 'r+') as file:
        data = file.read()
        data = re.sub(r'#.*', "", data)
        data = data.split()
    all = []
    column_names=['form_id', 'writer_id', 'num_of_sentences', 'word_segmentation', 'all_lines', 'lines_segmented', 'all_words', 'words_segmented']
    for idx in range(0, len(data), 8):
        row = {}
        row['form_id'] = data[idx]
        row['writer_id'] = data[idx + 1]
        row['num_of_sentences'] = int(data[idx + 2])
        row['word_segmentation'] = data[idx + 3]
        row['all_lines'] = int(data[idx + 4])
        row['lines_segmented'] = int(data[idx + 5])
        row['all_words'] = int(data[idx + 6])
        row['words_segmented'] = int(data[idx + 7])
        if set_image_path:
            row['image_path'] = ''.join(['data/forms', _get_file_str(row['form_id'][0]), '/', data[idx], '.png'])
            row['xml_path'] = ''.join(['data/xml/', data[idx], '.xml'])
        all.append(row)
    if set_image_path:
        column_names.append('image_path')
        column_names.append('xml_path')
    df = pd.DataFrame(all, columns=column_names)
    return df


def _get_file_str(letter):
    if letter <= 'd' and letter >= 'a':
        return forms_files_str[0]
    elif letter <= 'h':
        return forms_files_str[1]
    elif letter <= 'z':
        return forms_files_str[2]
    else:
        raise Exception


if __name__ == '__main__':
    df = read_forms_dataframe()
    # print(df['image_path'])
    sentences = df['all_words']
    plt.figure()
    sentences.plot.hist()
    plt.show()