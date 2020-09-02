from proverb_selector.sel_utils.file_manager import read_write_obj_file


def label_retrieval(filename):
    with open(filename, 'r') as label_file:
        l_reader = label_file.readlines()
        # proverb_reader = csv.reader(proverb_file)
        return [row.split(",") for c, row in enumerate(l_reader) if l_reader[c][0] != '%']


def label_processing(labels):
    list_labels = []
    for c, l in enumerate(labels):
        lemma, tmp = l[1].lower().strip().split(".")
        if tmp[0] == 'v':
            pos = tmp.split(":")
            pos.pop(0)
            list_labels.append((l[0], lemma, 'v', ":".join(pos)))
        elif ":" in tmp:
            tmp = tmp.split(":")
            pos = tmp[0]
            tmp.pop(0)
            form = ":".join(tmp)
            if '+' in pos:
                if 'letra' in pos:
                    continue
                pos = pos.split("+")[0]
            # (Original_Word, Lemma, PoS_Tag, Form)
            list_labels.append((l[0], lemma, pos, form))
        else:
            pos = form = tmp
            if '+' in tmp:
                pos = tmp.split("+")
                form = pos[1]
                pos = pos[0]
            list_labels.append((l[0], lemma, pos, form))
    print(len(list_labels))
    return list_labels


