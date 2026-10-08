"""MeetingMind - AI Meeting Minutes Generator (NLP mini project)
Run:  pip install -r requirements.txt  &&  streamlit run app.py
Each tab = one lab experiment, and its output feeds the final minutes (tab 10).
"""
import re, math
from collections import Counter
import numpy as np, pandas as pd, nltk, streamlit as st
from nltk import word_tokenize, sent_tokenize

st.set_page_config(page_title="MeetingMind", page_icon="📝", layout="wide")


@st.cache_resource(show_spinner="Downloading NLTK data (first run only)...")
def setup():
    for p in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger",
              "averaged_perceptron_tagger_eng", "maxent_ne_chunker", "maxent_ne_chunker_tab", "words", "treebank"]:
        nltk.download(p, quiet=True)
    return True


setup()
from nltk.corpus import stopwords, wordnet as wn, treebank, words as words_corpus
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.wsd import lesk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

STOP = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()
SAMPLE = """Priya: Good morning everyone. Let's start with the Q3 product launch plan.
Rahul: The launch is currently scheduled for October 25. The marketing team is ready, but the buget approval is still pending.
Priya: We have decided to move the launch date to November 5 so the team has more time to test the app.
Amit: I agree. The testing phase needs at least two more weeks.
Priya: Great. Amit will fix the remaining bugs in the login module by October 20.
Rahul: I will send the final budget report to the finance team by Friday.
Sneha: Should we also prepare a demo video for the customers?
Priya: Yes, Sneha needs to record the demo video before November 1.
Amit: One concern is the server capacity. We should discuss it in the next meeting.
Priya: We approved the new pricing plan after the discussion with the sales team.
Rahul: Priya will schedule the next meeting on Tuesday.
Sneha: The deadlne for the demo is tight, but the team is confident.
Priya: Thanks everyone. We finalized the plan and we will meet again next Tuesday."""

MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
DATE_RE = re.compile(
    rf"\b(?:(?:next|this|coming)\s+)?(?:{DAYS})\b|\b(?:{MONTHS})[a-z]*\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})[a-z]*|\btomorrow\b|\b(?:next|this)\s+(?:week|month)\b"
    rf"|\bend of (?:the )?(?:week|month|day)\b|\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b", re.I)
WHITELIST = {"login", "backend", "frontend", "demo", "app", "apps", "roadmap", "dashboard", "email", "emails", "online",
             "website", "api", "ui", "ux", "lakh", "okay", "feedback", "spreadsheet", "invoices", "agenda"}

# ---------------- Exp 03: Minimum Edit Distance ----------------
def med_matrix(a, b):
    a, b = a.lower(), b.lower()
    D = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1): D[i][0] = i
    for j in range(len(b) + 1): D[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return D


def med(a, b): return med_matrix(a, b)[-1][-1]


@st.cache_resource(show_spinner="Loading dictionary...")
def spell_res():
    freq = Counter(w.lower() for w in treebank.words())
    vocab = {w.lower() for w in words_corpus.words()} | set(freq) | WHITELIST
    byfirst = {}
    for w in vocab:
        if w.isalpha(): byfirst.setdefault(w[0], []).append(w)
    return vocab, freq, byfirst


VOCAB, FREQ, BYFIRST = spell_res()


def is_known(w): return w in VOCAB or w in STOP or bool(wn.morphy(w))


def correct(w):
    w = w.lower()
    if is_known(w): return w
    best = None
    for v in BYFIRST.get(w[0], []):
        if abs(len(v) - len(w)) <= 2:
            d = med(w, v)
            if d <= 2 and (best is None or (d, -FREQ[v]) < (best[0], -FREQ[best[1]])): best = (d, v)
    if not best: return None
    return best[1] if (best[0] <= 1 or len(w) >= 5) else None


def apply_fixes(text, fixes):
    if not fixes: return text
    pat = re.compile(r"\b(" + "|".join(map(re.escape, fixes)) + r")\b", re.I)
    return pat.sub(lambda m: fixes[m.group(0).lower()].capitalize() if m.group(0)[0].isupper() else fixes[m.group(0).lower()], text)


# ---------------- Exp 04: N-gram LM ----------------
def lm_build(seqs, n):
    cnt, ctx = Counter(), Counter()
    for s in seqs:
        p = ["<s>"] * (n - 1) + s + ["</s>"]
        for i in range(n - 1, len(p)):
            g = tuple(p[i - n + 1:i + 1]); cnt[g] += 1; ctx[g[:-1]] += 1
    return cnt, ctx


def lm_logp(model, seq, n, V):
    cnt, ctx = model
    p = ["<s>"] * (n - 1) + seq + ["</s>"]
    lp = 0.0
    for i in range(n - 1, len(p)):
        g = tuple(p[i - n + 1:i + 1]); lp += math.log((cnt[g] + 1) / (ctx[g[:-1]] + V))
    return lp, len(p) - (n - 1)


# ---------------- Exp 05: POS taggers ----------------
@st.cache_resource(show_spinner="Training POS taggers on Treebank...")
def taggers():
    sents = treebank.tagged_sents()
    tr, te = sents[:3000], sents[3000:]
    rules = nltk.RegexpTagger([(r"^(the|a|an)$", "DT"), (r"^-?\d+(\.\d+)?$", "CD"), (r".*ing$", "VBG"), (r".*ed$", "VBD"),
                               (r".*ould$", "MD"), (r".*\'s$", "POS"), (r".*s$", "NNS"), (r"^[A-Z].*", "NNP"), (r".*", "NN")])
    uni = nltk.UnigramTagger(tr, backoff=rules)
    bi = nltk.BigramTagger(tr, backoff=uni)
    acc = lambda t: t.accuracy(te) if hasattr(t, "accuracy") else t.evaluate(te)
    return rules, uni, bi, {"Rule-based (regex)": acc(rules), "Unigram + rules": acc(uni), "Bigram + unigram + rules": acc(bi)}


# ---------------- Exp 06: CFG parsing (top-down) ----------------
GRAMMAR_TXT = """S -> NP AUX VP | NP AUX VP PP | NP AUX RB VP | NP AUX RB VP PP
AUX -> MD | VB TO
VP -> VB NP | VB NP PP
PP -> IN DATE | IN NP | TO NP
NP -> NNP | NNP NNP | PRP | DT N | N | NNP N
N -> NN | JJ N | NN N"""
GRAMMAR_TXT = re.sub(r"\b(NNP|NN|PRP|DT|MD|TO|IN|RB|JJ|VB|DATE)\b", r"'\1'", GRAMMAR_TXT)  # POS tags are the terminals
GRAMMAR = nltk.CFG.fromstring(GRAMMAR_TXT)
PARSER = nltk.RecursiveDescentParser(GRAMMAR)


def coarse(w, t):
    if t == "DATE": return "DATE"
    if t.startswith("NNP"): return "NNP"
    if t.startswith("NN"): return "NN"
    if t in ("PRP", "DT", "MD", "TO", "IN", "RB"): return t
    if t.startswith("JJ") or t in ("VBG", "VBN", "CD", "PRP$"): return "JJ"
    if t.startswith("VB"): return "VB"
    return "X"


def cfg_parse(sent, speaker):
    s = re.sub(r"^(?:(?:yes|yeah|ok|okay|great|sure|right|so|also|and|well|but|then|thanks)\b[\s,.!]*)+", "", sent.strip(), flags=re.I)
    dates = [m.group(0) for m in DATE_RE.finditer(s)]
    tg = [(w, t) for w, t in nltk.pos_tag(word_tokenize(DATE_RE.sub("Friday", s))) if re.search(r"\w", w)]
    if not (4 <= len(tg) <= 22) or s.endswith("?"): return None
    toks, d = [], 0
    for w, t in tg:
        if w == "Friday" and d < len(dates): toks.append((dates[d], "DATE")); d += 1
        else: toks.append((w, t))
    seq = [coarse(w, t) for w, t in toks]
    try: trees = list(PARSER.parse(seq))
    except ValueError: return None
    if not trees: return None
    tr = trees[0]
    off = len(tr[0].leaves()) + len(tr[1].leaves())
    rest, rseq = toks[off:], seq[off:]
    when = next((w for (w, _), c in zip(rest, rseq) if c == "DATE"), "—")
    if "DATE" in rseq:
        k = rseq.index("DATE"); cut = k - 1 if k > 0 and rseq[k - 1] in ("IN", "TO") else k
        rest = rest[:cut] + rest[k + 1:]
    who = " ".join(w for w, _ in toks[:len(tr[0].leaves())])
    if who.lower() in ("i", "we", "you"): who = speaker
    return {"owner": who, "task": " ".join(w for w, _ in rest), "deadline": when, "tree": tr}


# ---------------- Exp 08: WordNet decision lexicon ----------------
SEEDS = ["decide", "agree", "approve", "finalize", "confirm", "conclude", "resolve"]
EXCL = {"support", "sustain", "find", "shape", "polish", "check", "name", "determine", "witness", "bear_witness", "affirm", "hold", "reason", "corroborate", "substantiate", "concord", "settle", "concur", "adjudicate", "o.k.", "okay", "sanction"}


@st.cache_resource
def decision_lexicon():
    lex = set(SEEDS)
    for s in SEEDS:
        for syn in wn.synsets(s, pos="v")[:1]:
            lex |= {l.lower() for l in syn.lemma_names() if "_" not in l and l.lower() not in EXCL}
    return lex


LEX = decision_lexicon()

# ---------------- Exp 09: classifier (self-created labelled data) ----------------
TRAIN = {
    "action": ["John will prepare the presentation by Friday.", "Maria needs to update the project plan before Monday.",
               "I will send the report to the client tomorrow.", "Ravi should finish the testing by October 12.",
               "Anita will book the conference room for next week.", "Please share the meeting notes with the team by Tuesday.",
               "Karan has to fix the login bug before the release.", "Neha will contact the vendor by the end of the week.",
               "We need to submit the final draft by November 3.", "Sam will review the budget and send feedback on Wednesday.",
               "Divya is going to schedule the demo for Thursday.", "Arun must complete the documentation before the deadline.",
               "Can you update the spreadsheet by tomorrow?", "Lisa will arrange the training session next month.",
               "Vikram should prepare the invoices by Monday.", "Pooja will call the supplier tomorrow.",
               "Rohit needs to clean the dataset before Friday.", "Tina will write the test cases by next week.",
               "I will share the slides after the meeting.", "Harsh has to deploy the model by Tuesday.",
               "Meera should send the minutes to everyone today."],
    "decision": ["We have decided to postpone the launch to December.", "The team agreed to use Python for the backend.",
                 "It was approved that the budget will be increased.", "We finalized the new pricing plan today.",
                 "Everyone agreed to hold the meeting every Monday.", "The committee decided to hire two more developers.",
                 "We confirmed the venue for the annual event.", "The proposal was approved by all members.",
                 "We have agreed to move the deadline to next week.", "It is decided that the app will support only Android first.",
                 "The board approved the new marketing strategy.", "We resolved to switch to the new vendor.",
                 "The group concluded that the second design is better.", "We agreed on a fixed budget of two lakh rupees.",
                 "Management approved the leave policy changes.", "We decided to cancel the second workshop.",
                 "The team agreed that testing will start in November.", "We have approved the revised timeline.",
                 "It was finalized that Ravi will lead the project.", "Everyone agreed to the new schedule.",
                 "We decided to go with the cloud option."],
    "discussion": ["The server capacity is a concern for the next release.", "Let us start with the sales numbers for this quarter.",
                   "How is the testing going so far?", "The customer feedback was mostly positive.",
                   "I think the design looks a bit crowded.", "We had some issues with the network last week.",
                   "Thanks everyone for joining the call.", "The marketing team is ready for the campaign.",
                   "What are the risks in this plan?", "Last quarter the revenue grew by ten percent.",
                   "I feel the interface could be improved.", "Good morning everyone, let us begin the meeting.",
                   "The competitor launched a similar product recently.", "There was a long discussion about the hiring plan.",
                   "The current system is slow when many users log in.",
                   "The testing phase needs at least two more weeks.", "The launch is currently scheduled for October 25.",
                   "The report is still pending approval.", "I agree.", "Great.", "The budget is tight this quarter.",
                   "Several users reported delays in the app.", "The deadline for the demo is tight but the team is confident."]}


@st.cache_resource(show_spinner="Training classifiers...")
def classifiers():
    X = [s for k, v in TRAIN.items() for s in v]
    y = [k for k, v in TRAIN.items() for _ in v]
    labels = list(TRAIN)
    models = {"Naive Bayes": make_pipeline(CountVectorizer(ngram_range=(1, 2)), MultinomialNB()),
              "Linear SVM": make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC())}
    res = {}
    for name, m in models.items():
        pred = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
        res[name] = {"acc": accuracy_score(y, pred), "cm": pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels),
                     "report": pd.DataFrame(classification_report(y, pred, output_dict=True)).T.round(2)}
        m.fit(X, y)
    best = max(res, key=lambda k: res[k]["acc"])
    return models, res, best


# ---------------- helpers ----------------
def wn_pos(tag): return {"J": "a", "V": "v", "N": "n", "R": "r"}.get(tag[0], "n")


def read_upload(f):
    n = f.name.lower()
    if n.endswith(".txt"): return f.read().decode("utf-8", "ignore")
    if n.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(f) as p: return "\n".join((pg.extract_text() or "") for pg in p.pages)
    if n.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(f).paragraphs)
    return ""


def parse_transcript(text):
    utts = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        m = re.match(r"^([A-Za-z][A-Za-z .]{0,25}):\s*(.+)$", line)
        utts.append((m.group(1).strip(), m.group(2)) if m else ("Unknown", line))
    return utts, [(sp, s) for sp, t in utts for s in sent_tokenize(t)]


def entities(tagged):
    try: tree = nltk.ne_chunk(tagged)
    except Exception: return []
    return [(" ".join(w for w, _ in t.leaves()), t.label()) for t in tree if hasattr(t, "label")]


def make_pdf(rep):
    from fpdf import FPDF
    pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(True, 15)
    clean = lambda t: str(t).encode("latin-1", "replace").decode("latin-1")
    def line(t, size=11, style=""):
        pdf.set_font("Helvetica", style, size); pdf.multi_cell(0, 6.5, clean(t), new_x="LMARGIN", new_y="NEXT")
    line("Meeting Minutes - MeetingMind", 17, "B"); pdf.ln(2)
    line("Participants", 13, "B"); line(", ".join(rep["participants"]) or "-")
    line("Summary", 13, "B")
    for s in rep["summary"]: line("- " + s)
    line("Key Decisions", 13, "B")
    for d in rep["decisions"] or ["None detected"]: line("- " + d)
    line("Action Items", 13, "B")
    for a in rep["actions"] or [{"owner": "-", "task": "None detected", "deadline": "-"}]:
        line(f"- {a['owner']}: {a['task']}  (Due: {a['deadline']})")
    line("Dates & Deadlines Mentioned", 13, "B")
    for d in rep["dates"] or ["None"]: line("- " + d)
    line("Keywords", 13, "B"); line(", ".join(rep["keywords"]))
    return bytes(pdf.output())


# =============== UI: input ===============
st.title("📝 MeetingMind - AI Meeting Minutes Generator")
st.caption("NLP mini project - every tab is one lab experiment (01-09) and feeds the final report (10).")
with st.sidebar:
    st.header("📥 Input transcript")
    up = st.file_uploader("Upload TXT / PDF / DOCX", type=["txt", "pdf", "docx"])
    raw = read_upload(up) if up else st.text_area("...or paste (format `Name: text`)", SAMPLE, height=320)
    fix_on = st.checkbox("Apply spelling correction (Exp 03)", True)
if not raw.strip(): st.info("Upload or paste a transcript to begin."); st.stop()

# Exp 01: script validation
letters = [c for c in raw if c.isalpha()]
latin = sum(c.isascii() for c in letters) / max(len(letters), 1)
if latin < 0.8: st.error(f"Script validation failed: only {latin:.0%} Latin letters. Please upload an English transcript."); st.stop()

utts0, _ = parse_transcript(raw)
speakers = {sp for sp, _ in utts0}
unknown = sorted({t.lower() for t in re.findall(r"[A-Za-z]+", raw) if len(t) >= 4 and not t[0].isupper() and not is_known(t.lower())})
fixes = {w: c for w in unknown if (c := correct(w)) and c != w}
text = apply_fixes(raw, fixes) if fix_on else raw
utts, sents = parse_transcript(text)
tagged = [nltk.pos_tag(word_tokenize(s)) for _, s in sents]
seqs = [[w.lower() for w, _ in tg if w.isalpha()] for tg in tagged]
ents = [entities(tg) for tg in tagged]

# classification / extraction (Exp 05-09 combined)
models, cres, best = classifiers()
rows, actions, decisions = [], [], []
for (sp, s), tg, seq in zip(sents, tagged, seqs):
    lem = [LEMM.lemmatize(w, "v") for w in seq]
    cfg = cfg_parse(s, sp)
    clf = models[best].predict([s])[0]
    if not s.endswith("?") and len(seq) >= 4 and set(lem) & LEX: label, via = "decision", "WordNet lexicon (08)"
    elif cfg: label, via = "action", "CFG parse (06)"
    else: label, via = clf, f"{best} (09)"
    if s.endswith("?") and label == "action": label = "discussion"  # questions are proposals, not assigned tasks
    rows.append({"Speaker": sp, "Sentence": s, "Label": label, "Decided by": via})
    if label == "decision": decisions.append(f"{s} ({sp})")
    if label == "action":
        if cfg: actions.append({"owner": cfg["owner"], "task": cfg["task"], "deadline": cfg["deadline"], "via": "CFG"})
        else:
            who = next((n for n in speakers if re.search(rf"\b{re.escape(n)}\b", s)), sp)
            m = DATE_RE.search(s)
            actions.append({"owner": who, "task": s, "deadline": m.group(0) if m else "—", "via": "classifier"})

content_nouns = Counter(LEMM.lemmatize(w.lower()) for tg in tagged for w, t in tg if t.startswith("NN") and w.lower() not in STOP and w.isalpha() and len(w) > 2 and w not in speakers)
cseqs = [[w for w in s if w not in STOP] for s in seqs]
bigrams = Counter(g for c in cseqs for g in zip(c, c[1:]))
keywords = [w for w, _ in content_nouns.most_common(8)] + [" ".join(g) for g, n in bigrams.most_common(4) if n >= 2]

texts = [s for _, s in sents]
if len(texts) > 3:
    M = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sc = np.asarray(M.sum(axis=1)).ravel() / np.sqrt([max(len(t.split()), 1) for t in texts])
    sc = np.where([len(t.split()) >= 6 for t in texts], sc, 0)
    summary = [texts[i] for i in sorted(np.argsort(sc)[-3:])]
else: summary = texts
date_list = [f"{m.group(0)}  -  \"{s}\"" for _, s in sents for m in DATE_RE.finditer(s)]
persons = sorted({e for el in ents for e, l in el if l == "PERSON" and e not in speakers and e.lower() not in STOP})
report = {"participants": sorted(speakers - {"Unknown"}), "summary": summary, "decisions": decisions,
          "actions": actions, "dates": date_list, "keywords": keywords}

tabs = st.tabs(["01 Preprocess", "02 Stem/Lemma", "03 Edit Distance", "04 N-grams", "05 POS", "06 CFG Parse",
                "07 NER/Chunk", "08 WSD/WordNet", "09 Classifier", "10 📄 Minutes"])

# ---- 01
with tabs[0]:
    st.subheader("Text preprocessing: tokenization, script validation, stopword removal")
    c1, c2, c3 = st.columns(3)
    c1.metric("Script validation", f"{latin:.0%} Latin", "valid")
    c2.metric("Sentences", len(sents)); c3.metric("Speakers", len(speakers))
    ex = [{"Sentence": s, "Tokens": " | ".join(w for w in word_tokenize(s)), "After stopword removal": " ".join(c)}
          for (_, s), c in list(zip(sents, cseqs))[:6]]
    st.dataframe(pd.DataFrame(ex), hide_index=True)
    st.caption(f"Stopwords removed overall: {sum(len(s) for s in seqs) - sum(len(c) for c in cseqs)} of {sum(len(s) for s in seqs)} word tokens.")

# ---- 02
with tabs[1]:
    st.subheader("Stemming vs Lemmatization")
    pos_of = {}
    for tg in tagged:
        for w, t in tg: pos_of.setdefault(w.lower(), t)
    top = [w for w, _ in Counter(w for c in cseqs for w in c).most_common(18)]
    P, L, S = PorterStemmer(), LancasterStemmer(), SnowballStemmer("english")
    df = pd.DataFrame([{"Word": w, "Porter": P.stem(w), "Lancaster": L.stem(w), "Snowball": S.stem(w),
                        "Lemma (WordNet)": LEMM.lemmatize(w, wn_pos(pos_of.get(w, "NN")))} for w in top])
    st.dataframe(df, hide_index=True)
    st.info("Stemmers chop suffixes and can produce non-words (e.g. 'budget' -> 'budget'/'budg' for aggressive ones); "
            "lemmatization returns real dictionary words, so MeetingMind uses **lemmas** for keywords and decision detection.")

# ---- 03
with tabs[2]:
    st.subheader("Minimum Edit Distance - spelling correction")
    st.write("**Typos found in this transcript:**")
    st.dataframe(pd.DataFrame([{"Typo": w, "Corrected to": fixes.get(w, "(no match)"), "Edit distance": med(w, fixes[w]) if w in fixes else None} for w in unknown]), hide_index=True) if unknown else st.success("No typos found.")
    TEST = [("buget", "budget"), ("deadlne", "deadline"), ("meetng", "meeting"), ("scedule", "schedule"), ("recieve", "receive"),
            ("projcet", "project"), ("decison", "decision"), ("managr", "manager"), ("reprot", "report"), ("fnal", "final")]
    ok = [correct(a) == b for a, b in TEST]
    st.write(f"**Evaluation on 10 sample misspellings: {sum(ok)}/10 correct ({sum(ok) * 10}%)**")
    st.dataframe(pd.DataFrame([{"Misspelled": a, "Expected": b, "Predicted": correct(a), "Correct": "✅" if k else "❌"} for (a, b), k in zip(TEST, ok)]), hide_index=True)
    c1, c2 = st.columns(2)
    w1, w2 = c1.text_input("Word 1", "intention"), c2.text_input("Word 2", "execution")
    if w1 and w2:
        D = med_matrix(w1, w2)
        st.write(f"Edit distance **{w1} -> {w2} = {D[-1][-1]}**")
        st.dataframe(pd.DataFrame(D, index=["#"] + [f"{c}{i + 1}" for i, c in enumerate(w1.lower())], columns=["#"] + [f"{c}{j + 1}" for j, c in enumerate(w2.lower())]))

# ---- 04
with tabs[3]:
    st.subheader("N-gram language models")
    sp_i = max(1, int(0.8 * len(seqs))); train, test = seqs[:sp_i], seqs[sp_i:] or seqs
    V = len({w for s in seqs for w in s}) + 2
    q = st.text_input("Sentence to score", texts[min(4, len(texts) - 1)])
    qs = [w.lower() for w in word_tokenize(q) if w.isalpha()]
    out = []
    for n, nm in [(1, "Unigram"), (2, "Bigram"), (3, "Trigram")]:
        m = lm_build(train, n)
        lp, _ = lm_logp(m, qs, n, V)
        tl = sum(lm_logp(m, s, n, V)[0] for s in test); tn = sum(lm_logp(m, s, n, V)[1] for s in test)
        out.append({"Model": nm, "P(sentence) [Laplace]": f"{math.exp(lp):.2e}", "Test perplexity": round(math.exp(-tl / tn), 1)})
    st.dataframe(pd.DataFrame(out), hide_index=True)
    st.caption("Train = first 80% sentences, test = last 20%. Higher-order models have higher perplexity on tiny data (sparsity) - expected.")
    tri = Counter(g for c in cseqs for g in zip(c, c[1:], c[2:]))
    c1, c2 = st.columns(2)
    c1.write("**Top key bigrams**"); c1.dataframe(pd.DataFrame([(" ".join(g), n) for g, n in bigrams.most_common(8)], columns=["Phrase", "Count"]), hide_index=True)
    c2.write("**Top key trigrams**"); c2.dataframe(pd.DataFrame([(" ".join(g), n) for g, n in tri.most_common(8)], columns=["Phrase", "Count"]), hide_index=True)

# ---- 05
with tabs[4]:
    st.subheader("POS tagging: rule-based vs probabilistic")
    rules, uni, bi, accs = taggers()
    st.write("**Accuracy on held-out Penn Treebank sentences (train 3000 / test 914):**")
    st.bar_chart(pd.Series(accs, name="accuracy"))
    st.dataframe(pd.DataFrame({"Tagger": accs.keys(), "Accuracy": [f"{v:.1%}" for v in accs.values()]}), hide_index=True)
    i = st.selectbox("Pick a sentence", range(len(texts)), format_func=lambda k: texts[k][:80])
    toks = [w for w, _ in tagged[i]]
    st.dataframe(pd.DataFrame({"Token": toks, "Rule-based": [t for _, t in rules.tag(toks)], "Bigram tagger": [t for _, t in bi.tag(toks)],
                               "Perceptron (used in pipeline)": [t for _, t in tagged[i]]}).T)
    st.caption("Pipeline uses the NLTK perceptron tagger for downstream steps (better context handling); MD + VB tag patterns signal action items.")

# ---- 06
with tabs[5]:
    st.subheader("Syntactic parsing with a small CFG (top-down recursive descent)")
    st.code(GRAMMAR_TXT, language="text")
    st.caption("Words are mapped to coarse POS tags (+ DATE from regex) and parsed top-down. The tree gives **who / what / when**.")
    shown = 0
    for (sp, s), r in zip(sents, rows):
        res = cfg_parse(s, sp)
        if res and shown < 4:
            shown += 1
            st.markdown(f"**\"{s}\"**  \n Owner: `{res['owner']}` | Task: `{res['task']}` | Deadline: `{res['deadline']}`")
            st.code(res["tree"].pformat(margin=90), language="text")
    own = st.text_input("Try your own sentence", "Ravi will update the project plan by Monday.")
    r = cfg_parse(own, "Speaker")
    st.json({"owner": r["owner"], "task": r["task"], "deadline": r["deadline"]}) if r else st.warning("Sentence does not fit the grammar (falls back to the classifier).")

# ---- 07
with tabs[6]:
    st.subheader("Named Entity Recognition & chunking")
    ent_df = pd.DataFrame([{"Entity": e, "Type": l} for el in ents for e, l in el]).drop_duplicates()
    st.write("**Named entities (NLTK ne_chunk)**"); st.dataframe(ent_df, hide_index=True) if len(ent_df) else st.write("None found.")
    st.write("**Dates / deadlines (regex)**"); st.write(date_list or "None")
    st.write(f"**Participants (speakers):** {', '.join(report['participants'])}" + (f"  \n**Other names mentioned:** {', '.join(persons)}" if persons else ""))
    chunker = nltk.RegexpParser("NP: {<DT>?<JJ.*>*<NN.*>+}")
    nps = Counter(" ".join(w for w, _ in t.leaves()).lower() for tg in tagged for t in chunker.parse(tg) if hasattr(t, "label") and len(t.leaves()) > 1)
    st.write("**Noun-phrase chunks = discussion topics**")
    st.dataframe(pd.DataFrame(nps.most_common(10), columns=["Noun phrase", "Count"]), hide_index=True)

# ---- 08
with tabs[7]:
    st.subheader("Word Sense Disambiguation (Lesk) + WordNet relations")
    out = []
    for w in [k for k, _ in content_nouns.most_common(12) if len(wn.synsets(k, "n")) > 1][:6]:
        i = next((j for j, s in enumerate(seqs) if w in [LEMM.lemmatize(x) for x in s]), None)
        if i is None: continue
        syn = lesk([x for x in word_tokenize(texts[i])], w, "n")
        out.append({"Word": w, "Context": texts[i][:70], "Sense chosen (Lesk)": syn.name() if syn else "-", "Definition": syn.definition() if syn else "-"})
    st.dataframe(pd.DataFrame(out), hide_index=True)
    word = st.text_input("Explore a word in WordNet", "good")
    syns = wn.synsets(word)
    if syns:
        c1, c2, c3 = st.columns(3)
        c1.write("**Synonyms**"); c1.write(", ".join(sorted({l.name() for s in syns for l in s.lemmas()})[:15]))
        c2.write("**Antonyms**"); c2.write(", ".join(sorted({a.name() for s in syns for l in s.lemmas() for a in l.antonyms()})) or "-")
        c3.write("**Hypernyms**"); c3.write(", ".join(sorted({h.name() for s in syns for h in s.hypernyms()})[:10]) or "-")
    st.write("**Decision-trigger lexicon (seed verbs expanded with WordNet synonyms)** - used to detect key decisions:")
    st.code(", ".join(sorted(LEX)), language="text")

# ---- 09
with tabs[8]:
    st.subheader("Sentence classifier: action item / decision / discussion")
    st.caption("Naive Bayes and SVM trained on 45 self-created labelled sentences (same technique as sentiment analysis, different labels). 5-fold cross-validation.")
    st.dataframe(pd.DataFrame({"Model": list(cres), "CV accuracy": [f"{r['acc']:.1%}" for r in cres.values()]}), hide_index=True)
    st.success(f"Model used in pipeline: **{best}**")
    c1, c2 = st.columns(2)
    c1.write("Confusion matrix"); c1.dataframe(cres[best]["cm"]); c2.write("Precision / Recall / F1"); c2.dataframe(cres[best]["report"])
    cs = st.text_input("Classify a sentence", "We agreed to launch on the first of December.")
    st.write(f"Prediction: **{models[best].predict([cs])[0]}**")
    st.write("**Transcript classified (final labels combine rules + classifier):**")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

# ---- 10
with tabs[9]:
    st.header("📄 Meeting Minutes")
    st.markdown("**Participants:** " + ", ".join(report["participants"]))
    st.markdown("#### Summary"); [st.markdown(f"- {s}") for s in summary]
    st.markdown("#### Key decisions"); [st.markdown(f"- {d}") for d in decisions] or st.write("None detected")
    st.markdown("#### Action items")
    if actions: st.dataframe(pd.DataFrame(actions).rename(columns=str.title), hide_index=True)
    else: st.write("None detected")
    st.markdown("#### Dates & deadlines"); [st.markdown(f"- {d}") for d in date_list]
    st.markdown("#### Keywords"); st.write(", ".join(keywords))
    st.download_button("⬇️ Download report as PDF", make_pdf(report), "meeting_minutes.pdf", "application/pdf")
