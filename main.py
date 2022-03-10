import sys
import json
from tqdm import tqdm
from spacy.tokens import DocBin
from spacy import blank


def parse_labels(input_labels):
    labels = []
    for label in json.load(input_labels):
        labels.append(label['text'])
    return labels


def convert(input_file_name, labels, output_file_name):
    spacy_json = {'classes': labels,
                  'annotations': []
                  }
                  
    with open(input_file_name, 'r', encoding='utf-8') as input_file:
        input_file_lines = input_file.readlines()
        for input_file_line in input_file_lines:
            json_object = json.loads(input_file_line)
            data = json_object['data']
            labels = json_object['label']
            annotations_metadata = [data, {'entities': labels}]
            spacy_json['annotations'].append(annotations_metadata)
        nlp = blank("en")
        db = DocBin()
        for text, annot in tqdm(spacy_json['annotations']):
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in annot["entities"]:
                span = doc.char_span(
                    start, end, label=label, alignment_mode="contract")
                if span is None:
                    print("Skipping entity")
                else:
                    ents.append(span)
            doc.ents = ents
            db.add(doc)

        db.to_disk(output_file_name)


def main():
    if len(sys.argv) == 4:
        input_file = sys.argv[1]
        input_labels = open(sys.argv[2])
        labels = parse_labels(input_labels)
        output_file = sys.argv[3]
        return convert(input_file, labels, output_file)
    print("Not enough arguments! Needs: Input JSONL, Input Labels, Output File Name.")


if __name__ == "__main__":
    main()
