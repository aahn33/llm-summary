import json

from pathlib import Path

# Recursively extracts text from a JSON file 
def extract_text(text):
    if isinstance(text, str): return text # Title of section
    elif isinstance(text, list): return '\n'.join(extract_text(section) for section in text) # Paragraphs of section
    
    res = ''
    for k, v in text.items():
        res += extract_text(v) + '\n'
    return res


fnames = open(Path('gov-report/split_ids/crs_test.ids')).read().split()

for fname in fnames[:50]:
    with open(Path(f'gov-report/crs/{fname}.json'), encoding='utf-8') as f:
        text = json.loads(f.read())
        
        d = {}
        d['title'] = text['title']
        d['summary'] = extract_text(text['summary'])
        d['full_text'] = extract_text(text['reports'])
        #print(d)

        Path('extracted').mkdir(exist_ok=True)
        with open(Path(f'extracted/{fname}.json'), 'w+', encoding='utf-8') as save:
            json.dump(d, save, ensure_ascii=False, indent=4)


