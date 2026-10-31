"""Name detector for TinyStories. Corpus-statistics: a capitalised word is a NAME if, over the corpus, it appears
capitalised in MID-sentence position >=MINC times and its lowercase form is (almost) never seen (cap ratio >= .97),
excluding I / weekdays / months / honorifics-only. Detector = whole-word regex over that name list (+ 'Mr./Mrs. X' handled as X).
Limits: sentence-initial-only names (never seen mid-sentence) missed; places like Paris counted as PER (type not separated)."""
import re, collections, json, sys
STOP = {"I","Mr","Mrs","Ms","Dr","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","January","February","March","April","May","June","July","August","September","October","November","December","God","Christmas","Easter","Halloween","Thank","Yes","No","Oh","Hello","Hi","Wow","Okay","OK","Let","Look","Please","Maybe","Sorry","Yay","Hey","Mom","Mommy","Dad","Daddy","Grandma","Grandpa","Mama","Papa","Santa"}
def stats(text):
    cap, low = collections.Counter(), collections.Counter()
    for m in re.finditer(r"(?<![\w'])([A-Za-z]+)(?![\w'])", text):
        w = m.group(1); s = m.start()
        if w[0].isupper():
            pre = text[max(0, s-2):s]
            if re.search(r'[.!?"\n]\s*$|^\s*$', pre) or s == 0 or text[max(0,s-1)] in '"\n' or re.search(r'[.!?]\s$', text[max(0,s-2):s]): continue
            cap[w] += 1
        else: low[w] += 1
    return cap, low
def build(text, minc=5, ratio=.97):
    cap, low = stats(text)
    return sorted(w for w, c in cap.items() if c >= minc and c / (c + low[w.lower()]) >= ratio and w not in STOP and len(w) > 1)
if __name__ == "__main__":
    t = open(sys.argv[1]).read().replace("<|endoftext|>", "\n\n")
    n = build(t); json.dump(n, open(sys.argv[2], "w")); print(len(n), n[:80])
