"""Text normalisation for business names and addresses (US, India, France and any other country label).

Canonicalisation maps every spelling of an abbreviation to one token (Corporation/Corp -> corp, Road/Rd -> rd,
Private/Pvt -> pvt ...). It is applied identically to both sides of every comparison, so collisions such as
Street/Saint -> st are harmless.
"""
import re
import unicodedata
import zlib

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# ---------------------------------------------------------------------------
# Indic script transliteration (Devanagari → Latin, plus other scripts via
# Unicode character names). Without this, Devanagari business names become
# empty strings and can never be blocked or matched.
# ---------------------------------------------------------------------------

_DEVA_CONSONANTS = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'f', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ळ': 'l', 'ऩ': 'n', 'ऱ': 'r',
}
_DEVA_VOWELS = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u', 'ऊ': 'uu',
    'ऋ': 'ri', 'ॠ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'ऑ': 'o',
}
_DEVA_MATRAS = {
    'ा': 'aa', 'ि': 'i', 'ी': 'ii', 'ु': 'u', 'ू': 'uu',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ॉ': 'o',
}
_DEVA_VIRAMA = '्'
_DEVA_ANUSVARA = 'ं'
_DEVA_CHANDRABINDU = 'ँ'
_DEVA_VISARGA = 'ः'
_DEVA_DIGITS = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}
_DEVA_SKIP = {'़', '‌', '‍', '।', '॥'}


def transliterate_devanagari(s):
    """Devanagari → Latin phonetic approximation. Non-Devanagari chars pass through unchanged."""
    if not any(0x0900 <= ord(c) <= 0x097F for c in s):
        return s
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in _DEVA_SKIP or c == _DEVA_VIRAMA:
            i += 1
            continue
        if c in _DEVA_DIGITS:
            out.append(_DEVA_DIGITS[c])
            i += 1
        elif c == _DEVA_ANUSVARA or c == _DEVA_CHANDRABINDU:
            out.append('n')
            i += 1
        elif c == _DEVA_VISARGA:
            out.append('h')
            i += 1
        elif c in _DEVA_VOWELS:
            out.append(_DEVA_VOWELS[c])
            i += 1
        elif c in _DEVA_CONSONANTS:
            cons = _DEVA_CONSONANTS[c]
            i += 1
            if i < len(s) and s[i] == _DEVA_VIRAMA:
                out.append(cons)          # no implicit vowel
                i += 1
            elif i < len(s) and s[i] in _DEVA_MATRAS:
                out.append(cons + _DEVA_MATRAS[s[i]])
                i += 1
            else:
                out.append(cons + 'a')    # implicit 'a'
            if i < len(s) and s[i] in (_DEVA_ANUSVARA, _DEVA_CHANDRABINDU):
                out.append('n')
                i += 1
        elif c in _DEVA_MATRAS:
            out.append(_DEVA_MATRAS[c])   # orphan matra
            i += 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


# Bengali (0x0980-0x09FF) — same structure as Devanagari; map consonants phonetically
_BENG_CONSONANTS = {
    'ক': 'k', 'খ': 'kh', 'গ': 'g', 'ঘ': 'gh', 'ঙ': 'ng',
    'চ': 'ch', 'ছ': 'chh', 'জ': 'j', 'ঝ': 'jh', 'ঞ': 'ny',
    'ট': 't', 'ঠ': 'th', 'ড': 'd', 'ঢ': 'dh', 'ণ': 'n',
    'ত': 't', 'থ': 'th', 'দ': 'd', 'ধ': 'dh', 'ন': 'n',
    'প': 'p', 'ফ': 'f', 'ব': 'b', 'ভ': 'bh', 'ম': 'm',
    'য': 'y', 'র': 'r', 'ল': 'l', 'শ': 'sh', 'ষ': 'sh', 'স': 's', 'হ': 'h',
    'ড়': 'r', 'ঢ়': 'rh', 'য়': 'y',
}
_BENG_VOWELS = {
    'অ': 'a', 'আ': 'aa', 'ই': 'i', 'ঈ': 'ii', 'উ': 'u', 'ঊ': 'uu',
    'এ': 'e', 'ঐ': 'ai', 'ও': 'o', 'ঔ': 'au',
}
_BENG_MATRAS = {
    'া': 'aa', 'ি': 'i', 'ী': 'ii', 'ু': 'u', 'ূ': 'uu',
    'ে': 'e', 'ৈ': 'ai', 'ো': 'o', 'ৌ': 'au',
}
_BENG_VIRAMA = '্'
_BENG_DIGITS = {
    '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
    '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9',
}


def transliterate_bengali(s):
    """Bengali → Latin phonetic approximation."""
    if not any(0x0980 <= ord(c) <= 0x09FF for c in s):
        return s
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in _BENG_DIGITS:
            out.append(_BENG_DIGITS[c])
            i += 1
        elif c == 'ঁ' or c == 'ং':   # anusvara/chandrabindu
            out.append('n')
            i += 1
        elif c == 'ঃ':   # visarga
            out.append('h')
            i += 1
        elif c in _BENG_VOWELS:
            out.append(_BENG_VOWELS[c])
            i += 1
        elif c in _BENG_CONSONANTS:
            cons = _BENG_CONSONANTS[c]
            i += 1
            if i < len(s) and s[i] == _BENG_VIRAMA:
                out.append(cons)
                i += 1
            elif i < len(s) and s[i] in _BENG_MATRAS:
                out.append(cons + _BENG_MATRAS[s[i]])
                i += 1
            else:
                out.append(cons + 'a')
        elif c in _BENG_MATRAS:
            out.append(_BENG_MATRAS[c])
            i += 1
        elif c == _BENG_VIRAMA:
            i += 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


# For Gujarati, Gurmukhi, Telugu, Kannada, Tamil, Malayalam — use Unicode character
# name heuristic: GUJARATI LETTER KA → extract "KA" as the phonetic value.
# This covers the most common case (standalone consonant with implicit 'a').
_INDIC_SCRIPT_RANGES = [
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0B80, 0x0BFF),  # Tamil
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0B00, 0x0B7F),  # Odia
]

_UNAME_CACHE = {}
_VOWEL_SIGN_RE = re.compile(r'^(?:\w+ )?(?:VOWEL SIGN|VOWEL|DIGIT) (.+)$')
_LETTER_RE = re.compile(r'^(?:\w+ )+LETTER ([A-Z]+)$')
_DIGIT_RE = re.compile(r'^(?:\w+ )+DIGIT (?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE)$')
_DIGIT_WORDS = {'ZERO': '0', 'ONE': '1', 'TWO': '2', 'THREE': '3', 'FOUR': '4',
                'FIVE': '5', 'SIX': '6', 'SEVEN': '7', 'EIGHT': '8', 'NINE': '9'}


def _indic_char_latin(c):
    """Map a single Indic script character to its approximate Latin phonetic value."""
    if c in _UNAME_CACHE:
        return _UNAME_CACHE[c]
    try:
        name = unicodedata.name(c, '')
    except Exception:
        name = ''
    result = c
    if not name:
        result = ' '
    else:
        parts = name.split()
        if 'DIGIT' in parts:
            dw = parts[-1]
            result = _DIGIT_WORDS.get(dw, ' ')
        elif 'VIRAMA' in parts or 'SIGN VIRAMA' in name:
            result = ''
        elif 'ANUSVARA' in parts or 'NUKTA' in parts:
            result = ''
        elif 'VISARGA' in parts:
            result = 'h'
        elif 'VOWEL SIGN' in name:
            # "LETTER AA" style
            m = re.search(r'VOWEL SIGN ([A-Z]+)$', name)
            result = m.group(1).lower() if m else 'a'
        elif 'LETTER' in parts:
            m = _LETTER_RE.match(name)
            result = (m.group(1).lower() + 'a') if m else ' '
        else:
            result = ' '
    _UNAME_CACHE[c] = result
    return result


def transliterate_indic(s):
    """Transliterate any Indic script character to Latin using Unicode character names."""
    if not any(any(lo <= ord(c) <= hi for lo, hi in _INDIC_SCRIPT_RANGES) for c in s):
        return s
    return ''.join(_indic_char_latin(c)
                   if any(lo <= ord(c) <= hi for lo, hi in _INDIC_SCRIPT_RANGES) else c
                   for c in s)


def transliterate_all(s):
    """Apply all Indic transliteration passes (Devanagari first, then Bengali, then others)."""
    s = transliterate_devanagari(s)
    s = transliterate_bengali(s)
    s = transliterate_indic(s)
    return s

NAME_MAP = {
    'corporation': 'corp', 'incorporated': 'inc', 'company': 'co', 'limited': 'ltd', 'private': 'pvt', 'pte': 'pvt',
    'international': 'intl', 'technologies': 'tech', 'technology': 'tech', 'svcs': 'services', 'svc': 'services',
    'brothers': 'bros', 'bro': 'bros', 'manufacturing': 'mfg', 'manufacturers': 'mfg', 'associates': 'assoc',
    'center': 'ctr', 'centre': 'ctr', 'saint': 'st', 'sainte': 'ste', 'mount': 'mt', 'fort': 'ft',
    'department': 'dept', 'hospital': 'hosp', 'industries': 'ind', 'industry': 'ind', 'inds': 'ind',
    'enterprises': 'ent', 'enterprise': 'ent', 'engineering': 'engg', 'eng': 'engg', 'medical': 'med',
    'pharmaceuticals': 'pharma', 'pharmaceutical': 'pharma', 'societe': 'ste', 'compagnie': 'cie',
    'etablissements': 'ets', 'establishment': 'ets',
    # Post-transliteration Hindi canonical forms
    'praivat': 'pvt', 'praivet': 'pvt', 'prayvet': 'pvt', 'praaivat': 'pvt',
    'limitid': 'ltd', 'limitied': 'ltd', 'limitad': 'ltd', 'limtied': 'ltd',
    'elaelapi': 'llp', 'laimitad': 'ltd', 'limieted': 'ltd',
    'korporeshana': 'corp', 'korporasan': 'corp',
    # Common Indian terms (romanized)
    'udyog': 'ind', 'udyam': 'ent', 'vyapar': 'trade', 'seva': 'services',
    'samiti': 'assoc', 'sansthan': 'inst', 'parishad': 'council',
    'bhandar': 'store', 'kendra': 'ctr', 'sanstha': 'inst',
    # Additional English abbreviations
    'solutions': 'soln', 'solution': 'soln', 'construction': 'const', 'constructions': 'const',
    'consultants': 'consult', 'consultant': 'consult', 'consulting': 'consult',
    'traders': 'trade', 'trading': 'trade', 'exports': 'exp', 'export': 'exp',
    'imports': 'imp', 'import': 'imp', 'services': 'svc', 'group': 'grp',
    'foundation': 'fdn', 'trust': 'trust', 'society': 'soc', 'cooperative': 'coop',
    'communication': 'comm', 'communications': 'comm', 'information': 'info',
    'financial': 'fin', 'finance': 'fin', 'investment': 'inv', 'investments': 'inv',
    'real': 'rl', 'estate': 'est', 'properties': 'prop', 'property': 'prop',
    'management': 'mgmt', 'development': 'dev', 'developer': 'dev',
    'global': 'gbl', 'national': 'natl', 'india': 'india',
    # French extras
    'societe': 'ste', 'association': 'assoc', 'syndicat': 'syn',
}
LEGAL = {
    'inc', 'corp', 'co', 'ltd', 'llc', 'llp', 'lp', 'pvt', 'plc', 'pllc', 'pc', 'opc', 'gmbh', 'ag', 'bv', 'nv', 'sa',
    'sas', 'sasu', 'sarl', 'eurl', 'snc', 'sci', 'scop', 'selarl', 'cie', 'srl', 'spa', 'pty', 'bhd', 'sdn', 'ltda',
}
NAME_STOP = {'the', 'and', 'of', 'de', 'du', 'des', 'la', 'le', 'les', 'et', 'l', 'd'}
ADDR_MAP = {
    'street': 'st', 'str': 'st', 'road': 'rd', 'avenue': 'ave', 'av': 'ave', 'boulevard': 'blvd', 'bd': 'blvd',
    'boul': 'blvd', 'lane': 'ln', 'drive': 'dr', 'place': 'pl', 'plaza': 'plz', 'court': 'ct', 'circle': 'cir',
    'suite': 'ste', 'apartment': 'apt', 'building': 'bldg', 'floor': 'fl', 'flr': 'fl', 'highway': 'hwy',
    'parkway': 'pkwy', 'expressway': 'expy', 'terrace': 'ter', 'square': 'sq', 'north': 'n', 'south': 's',
    'east': 'e', 'west': 'w', 'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw',
    'saint': 'st', 'sainte': 'ste', 'mount': 'mt', 'near': 'nr', 'opposite': 'opp', 'behind': 'bhnd',
    'sector': 'sec', 'sect': 'sec', 'nagar': 'ngr', 'colony': 'col', 'layout': 'lyt', 'extension': 'extn',
    'ext': 'extn', 'district': 'dist', 'dt': 'dist', 'village': 'vill', 'vlg': 'vill', 'number': 'no', 'num': 'no',
    'chemin': 'ch', 'allee': 'all', 'impasse': 'imp', 'route': 'rte', 'faubourg': 'fbg', 'quai': 'qu',
    'cours': 'crs', 'cross': 'crs', 'main': 'mn', 'ground': 'gnd', 'post': 'po',
    # US-specific
    'po': 'po', 'box': 'box', 'unit': 'unit', 'apt': 'apt', 'ste': 'ste', 'fl': 'fl',
    'freeway': 'fwy', 'blvd': 'blvd', 'ave': 'ave', 'dr': 'dr', 'ln': 'ln', 'ct': 'ct',
    # India-specific
    'plot': 'plt', 'khasra': 'khs', 'survey': 'srv', 'phase': 'ph', 'block': 'blk',
    'flat': 'flat', 'floor': 'fl', 'house': 'hno', 'door': 'dno', 'ward': 'ward',
    'taluka': 'tal', 'tehsil': 'teh', 'mandal': 'mdl', 'panchayat': 'pncht',
    'mohalla': 'mhl', 'gali': 'gali', 'chowk': 'cwk', 'marg': 'mrg',
    # France-specific
    'rue': 'rue', 'impasse': 'imp', 'residence': 'res', 'batiment': 'bat', 'bat': 'bat',
    'lotissement': 'lot', 'domaine': 'dom', 'hameau': 'ham', 'lieu': 'lieu',
    'cedex': 'cdx', 'bp': 'bp',
}
DBA_RE = re.compile(r'\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a\.?|dba|doing business as|trading as|t/a|a\.k\.a\.?|aka)\b', re.I)
LANDMARK_RE = re.compile(r'\b(?:near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of|close to|'
                         r'pres de|en face de|a cote de)\b')
_ELISION = re.compile(r"\b([ldjmnst]|qu)['']")          # French elision: l'atelier -> l atelier
_APOS = re.compile(r"[''`´]")
_ALNUM_SPLIT = re.compile(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])')   # sector5 -> sector 5
_NON_ALNUM = re.compile(r'[^a-z0-9]+')
_SKEL_SUBS = [('ksh', 'x'), ('ch', 'c'), ('sh', 's'), ('ph', 'f'), ('th', 't'), ('dh', 'd'), ('bh', 'b'),
              ('kh', 'k'), ('gh', 'g'), ('jh', 'j'), ('ck', 'k'), ('q', 'k'), ('w', 'v'), ('z', 's')]
_POSTAL_56 = re.compile(r'(?<!\d)(\d{5,6})(?!\d)')
_POSTAL_33 = re.compile(r'(?<!\d)(\d{3})\s+(\d{3})(?!\d)')
# URL/domain removal (www.foo.com, foo.com, http://...)
_URL_RE = re.compile(r'(?:https?://|www\.)\S+|\b\S+\.(?:com|org|net|in|fr|co)\b', re.I)
# Repeated-word collapse (VIDYALAYA VIDYALAYA → VIDYALAYA)
_WORD_DUP = re.compile(r'\b(\w+)( \1\b)+', re.I)


def strip_accents(s):
    """'Café Écolé' -> 'Cafe Ecole'."""
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def basic_norm(s):
    """Lower-case, strip accents, '&' -> 'and', drop apostrophes, split letter/digit runs, keep [a-z0-9 ].
    Indic scripts (Devanagari, Bengali, etc.) are transliterated to Latin before stripping."""
    s = _URL_RE.sub(' ', s)
    s = _WORD_DUP.sub(r'\1', s)
    s = transliterate_all(s)
    s = strip_accents(s).lower().replace('&', ' and ').replace('@', ' at ')
    s = _APOS.sub('', _ELISION.sub(r'\1 ', s))
    return _ALNUM_SPLIT.sub(' ', _NON_ALNUM.sub(' ', s)).strip()


def merge_single_letters(toks):
    """Join runs of single letters: ['l','l','c'] -> ['llc'], ['i','b','m'] -> ['ibm']."""
    out, buf = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append(''.join(buf))
            buf = []
        out.append(t)
    if buf:
        out.append(''.join(buf))
    return out


def canon_tokens(s, mapping):
    """Normalise a string and map every token through an abbreviation dictionary."""
    return [mapping.get(t, t) for t in merge_single_letters(basic_norm(s).split())]


def stem(t):
    """Crude plural stripping (traders -> trader)."""
    return t[:-1] if len(t) > 4 and t.endswith('s') and not t.endswith('ss') else t


def core_tokens(toks):
    """Name tokens without legal suffixes and stopwords (falls back to all tokens if nothing is left)."""
    c = [stem(t) for t in toks if t not in LEGAL and t not in NAME_STOP]
    return c if c else [stem(t) for t in toks]


def skeleton(toks):
    """Transliteration-tolerant key: merge digraphs, collapse repeats, drop inner vowels (Lakshmi ~ Laxmi)."""
    out = []
    for t in toks:
        if not t.isalpha():
            out.append(t)
            continue
        for a, b in _SKEL_SUBS:
            t = t.replace(a, b)
        t = re.sub(r'(.)\1+', r'\1', t)
        out.append(t[0] + re.sub(r'[aeiouyh]', '', t[1:]))
    return ' '.join(out)


NAME_COLS = ['n_norm', 'n_core', 'n_alt', 'n_legal', 'n_skel', 'n_ntok']


def process_name(raw):
    """Normalised views of one business name. n_alt is the DBA / trade-name alias core ('' if none)."""
    parts = DBA_RE.split(raw, maxsplit=1)
    main_t = canon_tokens(parts[0], NAME_MAP)
    alt_t = canon_tokens(parts[1], NAME_MAP) if len(parts) > 1 else []
    if not main_t and alt_t:
        main_t, alt_t = alt_t, []
    core_t = core_tokens(main_t)
    return (' '.join(main_t), ' '.join(core_t), ' '.join(core_tokens(alt_t)) if alt_t else '',
            ' '.join(sorted({t for t in main_t if t in LEGAL})), skeleton(core_t), len(core_t))


def extract_postal(a):
    """Last 5-6 digit number (US ZIP / India PIN / French code postal), also '700 032' PINs -> (code, parts)."""
    m = _POSTAL_56.findall(a)
    if m:
        return m[-1], {m[-1]}
    m = _POSTAL_33.findall(a)
    if m:
        return ''.join(m[-1]), set(m[-1])
    return '', set()


ADDR_COLS = ['a_norm', 'a_core', 'a_skel_unused', 'postal', 'nums', 'first_num', 'a_ntok', 'a_landmark']


def process_address(raw):
    """Normalised views of one address; landmark phrases ('Near SBI ATM') are cut out of a_core."""
    a = strip_accents(transliterate_all(raw))
    postal, postal_parts = extract_postal(a)
    core_segs, landmark = [], False
    for seg in re.split(r'[,;|\n]+', a.lower()):
        pieces = LANDMARK_RE.split(seg, maxsplit=1)
        landmark |= len(pieces) > 1
        core_segs.append(pieces[0])
    toks = canon_tokens(a, ADDR_MAP)
    core_t = canon_tokens(' , '.join(core_segs), ADDR_MAP)
    num_toks = [t for t in toks if t.isdigit() and t not in postal_parts]
    return (' '.join(toks), ' '.join(core_t), '', postal, ' '.join(sorted(set(num_toks))),
            num_toks[0] if num_toks else '', len(toks), landmark)


STR_COLS = ['n_core', 'n_alt', 'n_legal', 'n_skel', 'a_core', 'postal', 'nums', 'first_num']
N_HASH = 2 ** 22          # hashed key space for the blocking index


def _identity(x):
    """Analyzer for HashingVectorizer: documents are already lists of keys."""
    return x


_HV = HashingVectorizer(analyzer=_identity, n_features=N_HASH, alternate_sign=False, norm=None, binary=True,
                        dtype=np.float32)


def name_keys(core, skel, postal, fnum):
    """Blocking keys from the name: tokens, skeleton tokens, bigrams, compact name, and composite keys
    token@postal-code / token#house-number that stay rare even for very common names."""
    t = core.split()
    k = ['n:' + x for x in t] + ['s:' + x for x in skel.split()] + ['b:' + a + '_' + b for a, b in zip(t, t[1:])]
    if len(t) > 1:
        k.append('c:' + ''.join(t))
    if postal:
        k += ['p:' + x + '@' + postal for x in t]
    if fnum:
        k += ['h:' + x + '#' + fnum for x in t[:3]]
    return k


def addr_keys(acore, postal, fnum):
    """Blocking keys from the address: non-numeric tokens, bigrams, postal code, postal code + house number."""
    t = acore.split()
    k = ['a:' + x for x in t if not x.isdigit()] + ['g:' + a + '_' + b for a, b in zip(t, t[1:])]
    if postal:
        k.append('z:' + postal)
        if fnum:
            k.append('y:' + postal + '#' + fnum)
    return k


def encode_strs(strs):
    """List of str -> (utf-8 blob, int32 byte lengths) for compact storage."""
    b = [x.encode('utf-8') for x in strs]
    return b''.join(b), np.fromiter(map(len, b), np.int32, len(b))


def process_chunk(names, addrs):
    """Worker: normalise a chunk of records and build their hashed blocking keys.

    Returns (encoded text columns, numeric columns, name-key CSR structure, address-key CSR structure).
    """
    cols = {c: [] for c in STR_COLS}
    n_ntok, a_ntok, land, kn, ka = [], [], [], [], []
    for nm, ad in zip(names, addrs):
        _, core, alt, legal, skel, nt = process_name(nm)
        _, acore, _, postal, nums, fnum, at, lm = process_address(ad)
        for c, v in zip(STR_COLS, (core, alt, legal, skel, acore, postal, nums, fnum)):
            cols[c].append(v)
        n_ntok.append(nt)
        a_ntok.append(at)
        land.append(lm)
        kn.append(name_keys(core, skel, postal, fnum))
        ka.append(addr_keys(acore, postal, fnum))
    Xn, Xa = _HV.transform(kn), _HV.transform(ka)
    nums_out = {'n_ntok': np.array(n_ntok, np.int16), 'a_ntok': np.array(a_ntok, np.int16),
                'landmark': np.array(land, bool),
                'hn': np.array([zlib.crc32(x.encode()) for x in cols['n_core']], np.uint32),
                'ha': np.array([zlib.crc32(x.encode()) for x in cols['a_core']], np.uint32)}
    return ({c: encode_strs(v) for c, v in cols.items()}, nums_out,
            (Xn.indices.astype(np.int32), Xn.indptr.astype(np.int64)),
            (Xa.indices.astype(np.int32), Xa.indptr.astype(np.int64)))
