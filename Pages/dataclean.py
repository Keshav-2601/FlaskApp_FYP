# from dbm import error
# from idlelib.sidebar import EndLineDelegator
# from math import log10

import pandas as pd
import numpy as np
import json
import os
import re
# from IPython.utils.PyColorize import neutral_pygments_equiv
# from babel.messages.jslexer import hex_escape_re
# from pygments.lexer import words
PROJECTROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(PROJECTROOT, "CsvData", "News_Articles_Indian_Express.csv")
Indian_article_Data = pd.read_csv(csv_path)
Stops_words = [
        "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst",
        "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af",
        "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj",
        "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am",
        "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow",
        "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear",
        "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as",
        "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully",
        "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become",
        "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind",
        "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol",
        "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by",
        "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd",
        "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon",
        "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain",
        "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr",
        "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de",
        "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different",
        "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp",
        "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee",
        "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em",
        "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et",
        "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex",
        "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth",
        "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following",
        "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft",
        "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give",
        "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs",
        "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have",
        "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here",
        "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi",
        "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's",
        "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic",
        "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate",
        "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated",
        "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io",
        "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've",
        "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj",
        "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "latest", "lately", "later", "latter",
        "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked",
        "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt",
        "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means",
        "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml",
        "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must",
        "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near",
        "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless",
        "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless",
        "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny",
        "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj",
        "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or",
        "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside",
        "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par",
        "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj",
        "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp",
        "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly",
        "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2",
        "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs",
        "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively",
        "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs",
        "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd",
        "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen",
        "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan",
        "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've",
        "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar",
        "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody",
        "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon",
        "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop",
        "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy",
        "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten",
        "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the",
        "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered",
        "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon",
        "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin",
        "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three",
        "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together",
        "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts",
        "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un",
        "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur",
        "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value",
        "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt",
        "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome",
        "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever",
        "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas",
        "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim",
        "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why",
        "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont",
        "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi",
        "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd",
        "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt",
        "z", "zero", "zi", "zz",
    ]

Indian_article_Data
Indian_article_Data[['headline', 'desc', 'articles']]
Indian_article_Data = Indian_article_Data.drop(columns=['article_type', 'article_length'])
Indian_article_Data.columns
Indian_article_Data
# cleaning the stopped words from headline,desc,articles
special_Characters = {"!", "*", ".", "?", "%", "$", "'s", "(", ")", "#", "’s", ";", ","}

Stops_words = [w.lower() for w in Stops_words]


def remove_stopword(array_fo_string):
    finalcleanwords = []
    for word in array_fo_string:
        word = word.lower()
        if word not in Stops_words:
            finalcleanwords.append(word)
    return " ".join(finalcleanwords)


def clean_headline_text(incoming_string):
    if not isinstance(incoming_string, str) or pd.isna(incoming_string):
        return incoming_string
    words = incoming_string.split(" ")
    cleaned_words = []
    for word in words:
        word = word.lower()
        for ch in special_Characters:
            if (word.startswith(ch)):
                word = word.removeprefix(ch)
            elif (word.endswith(ch)):
                word = word.removesuffix(ch)
        cleaned_words.append(word)
    return remove_stopword(cleaned_words)


Indian_article_Data["Headline_Cleaned"] = Indian_article_Data["headline"].apply(clean_headline_text)


def clean_description_text(incoming_string):
    if not isinstance(incoming_string, str) or pd.isna(incoming_string):
        return incoming_string
    words = incoming_string.split(" ")
    cleaned_words = []
    for word in words:
        word = word.lower()
        for ch in special_Characters:
            if (word.startswith(ch)):
                word = word.removeprefix(ch)
            elif (word.endswith(ch)):
                word = word.removesuffix(ch)
        cleaned_words.append(word)
    return remove_stopword(cleaned_words)


Indian_article_Data["Desc_Cleaned"] = Indian_article_Data["desc"].apply(clean_description_text)

def clean_article_text(incoming_string):
    if not isinstance(incoming_string, str) or pd.isna(incoming_string):
        return incoming_string
    words = incoming_string.split(" ")
    cleaned_words = []
    for word in words:
        word = word.lower()
        for ch in special_Characters:
            if (word.startswith(ch)):
                word = word.removeprefix(ch)
            elif (word.endswith(ch)):
                word = word.removesuffix(ch)
        cleaned_words.append(word)
    return remove_stopword(cleaned_words)


Indian_article_Data["Article_Cleaned"] = Indian_article_Data["articles"].apply(clean_article_text)

Indian_article_Data.isnull().count()
Empty_Data = Indian_article_Data[
    Indian_article_Data.isnull().any(axis=1) | (Indian_article_Data.eq("").any(axis=1))]
Indian_article_Data
Indian_article_Data = Indian_article_Data.reset_index(drop=True)


##nice therefore there is no null data or Nan data.

def deepcleantext(required_row_data):
    special_characters = {"!", "*", ".", "?", "%", "$", "'s", "(", ")", "#", "’s", ";", ",", "-", "//", "^", "\\"}
    cleaned_text = ""
    numpy_array = required_row_data.values
    for row in numpy_array:
        row = row.lower()

        for ch in special_characters:
            row = row.replace(ch, " ")

        row = re.sub(r"[^a-z0-9\s]", " ", row)

        row = re.sub(r"\s+", " ", row).strip()

        cleaned_text += " " + row  # combine headline, desc, and article
    return cleaned_text


def getVsm(query1):
    ## take the Query
    ##cleaning the Query
    def tokenize(query1):
        words = query1.split(" ")

        print("original words ", words)
        cleaned_query = []
        for word in words:
            word = word.lower()
            if word not in Stops_words:
                for ch in special_Characters:
                    if (word.endswith(ch)):
                        word = word.removesuffix(ch)
                    elif (word.startswith(ch)):
                        word = word.removeprefix(ch)
                cleaned_query.append(word)
        return " ".join(cleaned_query)

    finalQuery = tokenize(query1)
    finalQuery

    length_of_properdata = []  # in order to  normalized each article.

    def calculatingTf():
        i = 0
        final_tf_array = []
        words_from_final_query = finalQuery.split(" ")
        while i < len(Indian_article_Data):
            Total_count_words = []
            required_row_data = Indian_article_Data.iloc[i][["Headline_Cleaned", "Desc_Cleaned", "Article_Cleaned"]]

            properdata = deepcleantext(
                required_row_data)  ##basically cleaning the data andCombining it in simple Large String

            length_of_properdata.append(len(properdata))
            for word in words_from_final_query:
                word = word.lower()
                occurence_of_word = len(re.findall(rf"\b{word}\b", properdata))

                Total_count_words.append(occurence_of_word)

            final_tf_array.append(Total_count_words)
            i = i + 1
        return (final_tf_array)

    final_tf_matrix = calculatingTf()
    # final_tf_matrix[1400]
    ##now to convert this into numpy array
    numpy_tf_score = np.array(final_tf_matrix)

    length_of_properdata  ## this is an array.
    numpy_norm_length_proper_data = np.array(
        length_of_properdata)  ## in oder to divide each elemnt of array with length
    length_reshape_in_order_to_divde = numpy_norm_length_proper_data.reshape(-1, 1)
    normalized_tf_score = numpy_tf_score / (
                length_reshape_in_order_to_divde + 1e-9)  # here 1e-9 means 10 power -9 to avoid division from zero.
    normalized_tf_score

    def calulateidf():
        i = 0
        words_from_final_query = finalQuery.split(" ")
        big_array_containing_all_idf_matrix = []
        while (i < len(Indian_article_Data)):
            idf_matrix_for_each_term = []
            mydata = Indian_article_Data.iloc[i][["Headline_Cleaned", "Desc_Cleaned", "Article_Cleaned"]]
            propercleandata = deepcleantext(mydata)
            for word in words_from_final_query:
                sum = 0
                word = word.lower()
                if re.search(rf"\b{re.escape(word)}\b", propercleandata):
                    sum += 1
                idf_matrix_for_each_term.append(sum)
            # print(idf_matrix_for_each_term)
            big_array_containing_all_idf_matrix.append(idf_matrix_for_each_term)
            i = i + 1
        return big_array_containing_all_idf_matrix

    # now vertically adding each number.

    my_idf_array = calulateidf()  # calling the function which is returning big array of data.

    idf_numpy_array = np.array(my_idf_array)
    final_idf_ans = idf_numpy_array.sum(axis=0)
    ##now log the weight to dapen the effect.
    N = len(Indian_article_Data)
    log_idf_ans = np.log10((N + 1) / (
                final_idf_ans + 1)) + 1  # adding 1 to avoid divide by zero thing # using np log for idf calculations.
    log_idf_ans

    numpy_tf_score.shape
    log_idf_ans.shape

    Document_tf_idf_score_vector = normalized_tf_score * log_idf_ans
    Document_tf_idf_score_vector

    finalQuery_array = finalQuery.split(" ")
    tf_query_matrix = []
    for word in finalQuery_array:
        word = word.lower()
        totalfrequencyofword = re.findall(rf"\b{(word)}\b", finalQuery) # gives array
        tf_query_matrix.append(len(totalfrequencyofword))

    tf_query_numpy_matrix = np.array(tf_query_matrix)
    print(tf_query_numpy_matrix)
    print(log_idf_ans)
    IDF_of_Query_vector = tf_query_numpy_matrix * log_idf_ans
    print(IDF_of_Query_vector)

    q = IDF_of_Query_vector.astype(float)
    D = Document_tf_idf_score_vector.astype(float)
    dot = D @ q  ## matrix muplitpilcation with q as d[i].q @ means dot product in python
    D_norm = np.linalg.norm(D, axis=1)  # mag of |d[i]|
    q_norm = np.linalg.norm(q)  # mag of q |q|

    den = D_norm * q_norm  # |d||q|
    cos_sims = np.divide(dot, den, out=np.zeros_like(dot), where=den != 0)  # cos0=(dot/|d|*|q|)
    angles_deg = np.degrees(np.arccos(np.clip(cos_sims, -1.0, 1.0)))  # 0=cos^-1(dot/|d|*|q|)

    best_doc_index = np.argmax(cos_sims)
    print("best doc index:", best_doc_index, "best similarity:", cos_sims[best_doc_index])
    data=Indian_article_Data[Indian_article_Data.index == best_doc_index].values[0]
    mydict={}
    for i in range(len(data)):
        mydict[i]=data[i]
    return json.dumps({"data":mydict})

def getLSA():
    Indian_article_Data = pd.read_csv("News_Articles_Indian_Express.csv")
    ## take the Query
    Stops_words = [
        "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst",
        "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af",
        "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj",
        "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am",
        "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow",
        "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear",
        "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as",
        "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully",
        "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become",
        "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind",
        "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol",
        "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by",
        "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd",
        "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon",
        "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain",
        "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr",
        "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de",
        "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different",
        "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp",
        "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee",
        "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em",
        "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et",
        "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex",
        "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth",
        "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following",
        "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft",
        "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give",
        "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs",
        "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have",
        "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here",
        "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi",
        "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's",
        "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic",
        "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate",
        "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated",
        "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io",
        "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've",
        "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj",
        "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "latest", "lately", "later", "latter",
        "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked",
        "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt",
        "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means",
        "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml",
        "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must",
        "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near",
        "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless",
        "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless",
        "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny",
        "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj",
        "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or",
        "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside",
        "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par",
        "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj",
        "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp",
        "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly",
        "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2",
        "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs",
        "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively",
        "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs",
        "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd",
        "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen",
        "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan",
        "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've",
        "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar",
        "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody",
        "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon",
        "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop",
        "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy",
        "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten",
        "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the",
        "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered",
        "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon",
        "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin",
        "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three",
        "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together",
        "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts",
        "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un",
        "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur",
        "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value",
        "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt",
        "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome",
        "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever",
        "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas",
        "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim",
        "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why",
        "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont",
        "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi",
        "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd",
        "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt",
        "z", "zero", "zi", "zz",
    ]

    Indian_article_Data
    Indian_article_Data[['headline', 'desc', 'articles']]

    Indian_article_Data = Indian_article_Data.drop(columns=['article_type', 'article_length'])
    Indian_article_Data.columns

    Indian_article_Data
    # cleaning the stopped words from headline,desc,articles

    special_Characters = {"!", "*", ".", "?", "%", "$", "'s", "(", ")", "#", "’s", ";", ","}
    Stops_words = [w.lower() for w in Stops_words]

    def remove_stopword(array_fo_string):
        finalcleanwords = []
        for word in array_fo_string:
            word = word.lower()
            if word not in Stops_words:
                finalcleanwords.append(word)
        return " ".join(finalcleanwords)

    def clean_headline_text(incoming_string):
        if not isinstance(incoming_string, str) or pd.isna(incoming_string):
            return incoming_string
        words = incoming_string.split(" ")
        cleaned_words = []
        for word in words:
            word = word.lower()
            for ch in special_Characters:
                if (word.startswith(ch)):
                    word = word.removeprefix(ch)
                elif (word.endswith(ch)):
                    word = word.removesuffix(ch)
            cleaned_words.append(word)
        return remove_stopword(cleaned_words)

    Indian_article_Data["Headline_Cleaned"] = Indian_article_Data["headline"].apply(clean_headline_text)

    def clean_description_text(incoming_string):
        if not isinstance(incoming_string, str) or pd.isna(incoming_string):
            return incoming_string
        words = incoming_string.split(" ")
        cleaned_words = []
        for word in words:
            word = word.lower()
            for ch in special_Characters:
                if (word.startswith(ch)):
                    word = word.removeprefix(ch)
                elif (word.endswith(ch)):
                    word = word.removesuffix(ch)
            cleaned_words.append(word)
        return remove_stopword(cleaned_words)

    Indian_article_Data["Desc_Cleaned"] = Indian_article_Data["desc"].apply(clean_description_text)

    def clean_description_text(incoming_string):
        if not isinstance(incoming_string, str) or pd.isna(incoming_string):
            return incoming_string
        words = incoming_string.split(" ")
        cleaned_words = []
        for word in words:
            word = word.lower()
            for ch in special_Characters:
                if (word.startswith(ch)):
                    word = word.removeprefix(ch)
                elif (word.endswith(ch)):
                    word = word.removesuffix(ch)
            cleaned_words.append(word)
        return remove_stopword(cleaned_words)

    Indian_article_Data["Desc_Cleaned"] = Indian_article_Data["desc"].apply(clean_description_text)

    def clean_article_text(incoming_string):
        if not isinstance(incoming_string, str) or pd.isna(incoming_string):
            return incoming_string
        words = incoming_string.split(" ")
        cleaned_words = []
        for word in words:
            word = word.lower()
            for ch in special_Characters:
                if (word.startswith(ch)):
                    word = word.removeprefix(ch)
                elif (word.endswith(ch)):
                    word = word.removesuffix(ch)
            cleaned_words.append(word)
        return remove_stopword(cleaned_words)

    Indian_article_Data["Article_Cleaned"] = Indian_article_Data["articles"].apply(clean_article_text)

    Indian_article_Data.isnull().count()
    Empty_Data = Indian_article_Data[
        Indian_article_Data.isnull().any(axis=1) | (Indian_article_Data.eq("").any(axis=1))]
    Indian_article_Data
    Indian_article_Data = Indian_article_Data.reset_index(drop=True)

    ##nice therefore there is no null data or Nan data.

    def deepcleantext(required_row_data):
        special_characters = {"!", "*", ".", "?", "%", "$", "'s", "(", ")", "#", "’s", ";", ",", "-", "//", "^", "\\"}
        cleaned_text = ""
        numpy_array = required_row_data.values
        for row in numpy_array:
            row = row.lower()

            for ch in special_characters:
                row = row.replace(ch, " ")

            row = re.sub(r"[^a-z0-9\s]", " ", row)

            row = re.sub(r"\s+", " ", row).strip()

            cleaned_text += " " + row  # combine headline, desc, and article
        return cleaned_text

    query1 = "Odisha trainer aircraft crash how many people died"
    query2 = "Hizbul militants killed in shopian encounter"
    query3 = "Mamata banerjee opposition virtual meet on covid lockdown"
    query4 = "Unlockdown sops 50 percent seating restaurants malls masks distancing"
    query5 = "Gujarat gang dressed as women looting truck drivers on highways"
    query6 = "Telangana covid deaths senior citizens dme advice treat every patient as covid suspect"
    query7 = "Madhya pradesh wheat procurement 125 lmt surpass punjab central pool"
    query8 = "West bengal government converting schools and tourist lodges into covid hospitals"
    query9 = "Why sanjay raut accused sonu sood of doing political script for bjp"
    query10 = "Militants killed in encounter with security forces in shopian south kashmir"
    query11 = "Tiger attacks and kills people in chandrapur near tadoba andhari"
    query12 = "India china ladakh lac standoff mea statement on resolving border tensions by bilateral agreements"
    query13 = "karnataka highest single day spike 99 covid cases bengaluru"
    query14 = "kartarpur corridor inaugurated modi imran khan 562 pilgrims"
    query15 = "west bengal police stop dilip ghosh amphan hit east midnapore relief permit"
    query16 = "mobile internet restored kargil 145 days shutdown after article 370"
    query17 = "What is the latest news on the India-China border tension and defense talks?"

    ##cleaning the Query
    def tokenize(query1):
        words = query1.split(" ")

        print("original words ", words)
        cleaned_query = []
        for word in words:
            word = word.lower()
            if word not in Stops_words:
                for ch in special_Characters:
                    if (word.endswith(ch)):
                        word = word.removesuffix(ch)
                    elif (word.startswith(ch)):
                        word = word.removeprefix(ch)
                cleaned_query.append(word)
        return " ".join(cleaned_query)

    finalQuery = tokenize(query10)
    finalQuery

    length_of_properdata = []  # in order to  normalized each article.

    def calculatingTf():
        i = 0
        final_tf_array = []
        words_from_final_query = finalQuery.split(" ")
        while i < len(Indian_article_Data):
            Total_count_words = []
            required_row_data = Indian_article_Data.iloc[i][["Headline_Cleaned", "Desc_Cleaned", "Article_Cleaned"]]

            properdata = deepcleantext(
                required_row_data)  ##basically cleaning the data andCombining it in simple Large String

            # lemmizationData(properdata)

            length_of_properdata.append(len(properdata))
            for word in words_from_final_query:
                word = word.lower()
                occurence_of_word = len(re.findall(rf"\b{word}\b", properdata))

                Total_count_words.append(occurence_of_word)

            final_tf_array.append(Total_count_words)
            i = i + 1
        return (final_tf_array)

    final_tf_matrix = calculatingTf()
    final_tf_matrix[1400]
    ##now to convert this into numpy array
    numpy_tf_score = np.array(final_tf_matrix)

    length_of_properdata  ## this is an array.
    numpy_norm_length_proper_data = np.array(
        length_of_properdata)  ## in oder to divide each elemnt of array with length
    length_reshape_in_order_to_divde = numpy_norm_length_proper_data.reshape(-1, 1)
    normalized_tf_score = numpy_tf_score / (
                length_reshape_in_order_to_divde + 1e-9)  # here 1e-9 means 10 power -9 to avoid division from zero.
    normalized_tf_score

    def calulateidf():
        i = 0
        words_from_final_query = finalQuery.split(" ")
        big_array_containing_all_idf_matrix = []
        while (i < len(Indian_article_Data)):
            idf_matrix_for_each_term = []
            mydata = Indian_article_Data.iloc[i][["Headline_Cleaned", "Desc_Cleaned", "Article_Cleaned"]]
            propercleandata = deepcleantext(mydata)
            for word in words_from_final_query:
                sum = 0
                word = word.lower()
                if re.search(rf"\b{re.escape(word)}\b", propercleandata):
                    sum += 1
                idf_matrix_for_each_term.append(sum)
            # print(idf_matrix_for_each_term)
            big_array_containing_all_idf_matrix.append(idf_matrix_for_each_term)
            i = i + 1
        return big_array_containing_all_idf_matrix

    # now vertically adding each number.

    my_idf_array = calulateidf()  # calling the function which is returning big array of data.
    idf_numpy_array = np.array(my_idf_array)
    final_idf_ans = idf_numpy_array.sum(axis=0)
    ##now log the weight to dapen the effect.
    N = len(Indian_article_Data)
    log_idf_ans = np.log10((N + 1) / (
                final_idf_ans + 1)) + 1  # adding 1 to avoid divide by zero thing # using np log for idf calculations.
    log_idf_ans

    numpy_tf_score.shape
    log_idf_ans.shape

    Document_tf_idf_score_vector = normalized_tf_score * log_idf_ans
    Document_tf_idf_score_vector

    finalQuery_array = finalQuery.split(" ")
    tf_query_matrix = []
    for word in finalQuery_array:
        word = word.lower()
        totalfrequencyofword = re.findall(rf"\b{(word)}\b", finalQuery)
        tf_query_matrix.append(len(totalfrequencyofword))

    tf_query_numpy_matrix = np.array(tf_query_matrix)
    print(tf_query_numpy_matrix)
    print(log_idf_ans)
    IDF_of_Query_vector = tf_query_numpy_matrix * log_idf_ans
    print(IDF_of_Query_vector)

    q = IDF_of_Query_vector.astype(float)
    D = Document_tf_idf_score_vector.astype(float)
    dot = D @ q  ## matrix muplitpilcation with q as d[i].q @ means dot product in python
    D_norm = np.linalg.norm(D, axis=1)  # mag of |d[i]|
    q_norm = np.linalg.norm(q)  # mag of q |q|

    den = D_norm * q_norm  # |d||q|
    cos_sims = np.divide(dot, den, out=np.zeros_like(dot), where=den != 0)  # cos0=(dot/|d|*|q|)
    angles_deg = np.degrees(np.arccos(np.clip(cos_sims, -1.0, 1.0)))  # 0=cos^-1(dot/|d|*|q|)

    best_doc_index = np.argmax(cos_sims)
    print("best doc index:", best_doc_index, "best similarity:", cos_sims[best_doc_index])
    Indian_article_Data[Indian_article_Data.index == best_doc_index].values[0]

    ##LSA Starting.(Latent Semantics Analysis)
    '''
    now I want to minimize the corpus of document using SVD .
    Matrix Corpus of Document will be SVD into |D|=UΣV^T
    Σ= is the diagonal matrix of A^TA sq.root of eigenvalues first find the ATA
    '''
    # My A vector is document corpus .
    A = Document_tf_idf_score_vector
    print(type(A))  # yes it's a numpy matrix

    A_transpose = np.transpose(A)
    print(A_transpose.shape)
    ## now A _tranpose * A
    ATA = A_transpose @ A
    ATA.shape
    # now i have ot find eigenvalues of ATA this is my square matrix so it will statify eq
    eigen_value = np.linalg.eigh(ATA)
    # now singluar values will be root of all eigen value
    singular_value = np.sqrt(eigen_value.eigenvalues)
    singular_value.sort()
    singular_value = singular_value[::-1]
    # therefore Σ or sigma =singular value diagonal matrix in descedning order.
    singular_value
    sigma_diagonal = np.diag(singular_value)  # Σ or sigma
    sigma_diagonal

    # now we have to find U or VT
    # V is the eigenvector of ATA from the spectrum theorem
    V = eigen_value.eigenvectors
    print(V)
    V.shape  # it's the orthogonal matrix.
    V_transpose = np.transpose(V)
    V_transpose
    return "LSA ANS"
#######################################################################################
#################################

#######################################################################################
#################################


#######################################################################################
#################################


#######################################################################################
#################################



## Original Code from Jupyter File

# Indian_article_Data=pd.read_csv("News_Articles_Indian_Express.csv")
# ## take the Query
# Stops_words=[
#     "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "latest","lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz",
# ]
#
# Indian_article_Data
# Indian_article_Data[['headline','desc','articles']]
#
# Indian_article_Data=Indian_article_Data.drop(columns=['article_type', 'article_length'])
# Indian_article_Data.columns
#
# Indian_article_Data
# #cleaning the stopped words from headline,desc,articles
#
#
#
# special_Characters = {"!", "*", ".", "?", "%", "$", "'s", "(", ")", "#", "’s", ";", ","}
# Stops_words = [w.lower() for w in Stops_words]
#
# def remove_stopword(array_fo_string):
#     finalcleanwords=[]
#     for word in array_fo_string:
#         word=word.lower()
#         if word not in Stops_words:
#             finalcleanwords.append(word)
#     return " ".join(finalcleanwords)
#
#
# def clean_headline_text(incoming_string):
#     if not isinstance(incoming_string,str) or pd.isna(incoming_string):
#         return incoming_string
#     words = incoming_string.split(" ")
#     cleaned_words = []
#     for word in words:
#         word=word.lower()
#         for ch in special_Characters:
#             if(word.startswith(ch)):
#                 word=word.removeprefix(ch)
#             elif(word.endswith(ch)):
#                 word=word.removesuffix(ch)
#         cleaned_words.append(word)
#     return remove_stopword(cleaned_words)
# Indian_article_Data["Headline_Cleaned"] = Indian_article_Data["headline"].apply(clean_headline_text)
#
# def clean_description_text(incoming_string):
#     if not isinstance(incoming_string,str) or pd.isna(incoming_string):
#         return incoming_string
#     words = incoming_string.split(" ")
#     cleaned_words = []
#     for word in words:
#         word=word.lower()
#         for ch in special_Characters:
#             if(word.startswith(ch)):
#                 word=word.removeprefix(ch)
#             elif(word.endswith(ch)):
#                 word=word.removesuffix(ch)
#         cleaned_words.append(word)
#     return remove_stopword(cleaned_words)
# Indian_article_Data["Desc_Cleaned"] = Indian_article_Data["desc"].apply(clean_description_text)
#
#
# def clean_description_text(incoming_string):
#     if not isinstance(incoming_string,str) or pd.isna(incoming_string):
#         return incoming_string
#     words = incoming_string.split(" ")
#     cleaned_words = []
#     for word in words:
#         word=word.lower()
#         for ch in special_Characters:
#             if(word.startswith(ch)):
#                 word=word.removeprefix(ch)
#             elif(word.endswith(ch)):
#                 word=word.removesuffix(ch)
#         cleaned_words.append(word)
#     return remove_stopword(cleaned_words)
# Indian_article_Data["Desc_Cleaned"] = Indian_article_Data["desc"].apply(clean_description_text)
#
# def clean_article_text(incoming_string):
#     if not isinstance(incoming_string,str) or pd.isna(incoming_string):
#         return incoming_string
#     words = incoming_string.split(" ")
#     cleaned_words = []
#     for word in words:
#         word=word.lower()
#         for ch in special_Characters:
#             if(word.startswith(ch)):
#                 word=word.removeprefix(ch)
#             elif(word.endswith(ch)):
#                 word=word.removesuffix(ch)
#         cleaned_words.append(word)
#     return remove_stopword(cleaned_words)
# Indian_article_Data["Article_Cleaned"] = Indian_article_Data["articles"].apply(clean_article_text)
#
#
# Indian_article_Data.isnull().count()
# Empty_Data=Indian_article_Data[Indian_article_Data.isnull().any(axis=1) | (Indian_article_Data.eq("").any(axis=1))]
# Indian_article_Data
# Indian_article_Data=Indian_article_Data.reset_index(drop=True)
# ##nice therefore there is no null data or Nan data.
#
# def deepcleantext(required_row_data):
#     special_characters={"!", "*", ".", "?", "%", "$", "'s", "(", ")", "#", "’s", ";", ",","-","//","^","\\"}
#     cleaned_text = ""
#     numpy_array = required_row_data.values
#     for row in numpy_array:
#         row = row.lower()
#
#         for ch in special_characters:
#             row = row.replace(ch, " ")
#
#         row = re.sub(r"[^a-z0-9\s]", " ", row)
#
#         row = re.sub(r"\s+", " ", row).strip()
#
#         cleaned_text += " " + row  # combine headline, desc, and article
#     return cleaned_text
#
#
#
# query1  = "Odisha trainer aircraft crash how many people died"
# query2  = "Hizbul militants killed in shopian encounter"
# query3  = "Mamata banerjee opposition virtual meet on covid lockdown"
# query4  = "Unlockdown sops 50 percent seating restaurants malls masks distancing"
# query5  = "Gujarat gang dressed as women looting truck drivers on highways"
# query6  = "Telangana covid deaths senior citizens dme advice treat every patient as covid suspect"
# query7  = "Madhya pradesh wheat procurement 125 lmt surpass punjab central pool"
# query8  = "West bengal government converting schools and tourist lodges into covid hospitals"
# query9  = "Why sanjay raut accused sonu sood of doing political script for bjp"
# query10 = "Militants killed in encounter with security forces in shopian south kashmir"
# query11 = "Tiger attacks and kills people in chandrapur near tadoba andhari"
# query12 = "India china ladakh lac standoff mea statement on resolving border tensions by bilateral agreements"
# query13="karnataka highest single day spike 99 covid cases bengaluru"
# query14="kartarpur corridor inaugurated modi imran khan 562 pilgrims"
# query15="west bengal police stop dilip ghosh amphan hit east midnapore relief permit"
# query16="mobile internet restored kargil 145 days shutdown after article 370"
# query17="What is the latest news on the India-China border tension and defense talks?"
# ##cleaning the Query
# def tokenize(query1):
#     words=query1.split(" ")
#
#     print("original words ",words)
#     cleaned_query=[]
#     for word in words:
#         word=word.lower()
#         if word not in Stops_words:
#             for ch in special_Characters:
#                 if(word.endswith(ch)):
#                     word=word.removesuffix(ch)
#                 elif(word.startswith(ch)):
#                     word=word.removeprefix(ch)
#             cleaned_query.append(word)
#     return " ".join(cleaned_query)
#
# finalQuery=tokenize(query10)
# finalQuery
#
#
# length_of_properdata=[] # in order to  normalized each article.
# def calculatingTf():
#     i=0
#     final_tf_array=[]
#     words_from_final_query=finalQuery.split(" ")
#     while i<len(Indian_article_Data):
#       Total_count_words=[]
#       required_row_data = Indian_article_Data.iloc[i][["Headline_Cleaned","Desc_Cleaned","Article_Cleaned"]]
#
#       properdata=deepcleantext(required_row_data)##basically cleaning the data andCombining it in simple Large String
#
#       #lemmizationData(properdata)
#
#       length_of_properdata.append(len(properdata))
#       for word in words_from_final_query:
#         word=word.lower()
#         occurence_of_word=len(re.findall(rf"\b{word}\b", properdata))
#
#         Total_count_words.append(occurence_of_word)
#
#       final_tf_array.append(Total_count_words)
#       i=i+1
#     return(final_tf_array)
# final_tf_matrix=calculatingTf()
# final_tf_matrix[1400]
# ##now to convert this into numpy array
# numpy_tf_score=np.array(final_tf_matrix)
#
#
#
#
# length_of_properdata ## this is an array.
# numpy_norm_length_proper_data=np.array(length_of_properdata) ## in oder to divide each elemnt of array with length
# length_reshape_in_order_to_divde=numpy_norm_length_proper_data.reshape(-1,1)
# normalized_tf_score=numpy_tf_score/(length_reshape_in_order_to_divde+ 1e-9) #here 1e-9 means 10 power -9 to avoid division from zero.
# normalized_tf_score
#
#
#
# def calulateidf():
#     i=0
#     words_from_final_query=finalQuery.split(" ")
#     big_array_containing_all_idf_matrix=[]
#     while(i<len(Indian_article_Data)):
#         idf_matrix_for_each_term=[]
#         mydata= Indian_article_Data.iloc[i][["Headline_Cleaned","Desc_Cleaned","Article_Cleaned"]]
#         propercleandata=deepcleantext(mydata)
#         for word in words_from_final_query:
#           sum=0
#           word=word.lower()
#           if re.search(rf"\b{re.escape(word)}\b", propercleandata):
#                 sum += 1
#           idf_matrix_for_each_term.append(sum)
#         # print(idf_matrix_for_each_term)
#         big_array_containing_all_idf_matrix.append(idf_matrix_for_each_term)
#         i=i+1
#     return big_array_containing_all_idf_matrix
# #now vertically adding each number.
#
# my_idf_array=calulateidf() # calling the function which is returning big array of data.
# idf_numpy_array=np.array(my_idf_array)
# final_idf_ans=idf_numpy_array.sum(axis=0)
# ##now log the weight to dapen the effect.
# N = len(Indian_article_Data)
# log_idf_ans = np.log10((N + 1) / (final_idf_ans + 1)) + 1 # adding 1 to avoid divide by zero thing # using np log for idf calculations.
# log_idf_ans
#
#
# numpy_tf_score.shape
# log_idf_ans.shape
#
#
# Document_tf_idf_score_vector=normalized_tf_score*log_idf_ans
# Document_tf_idf_score_vector
#
#
# finalQuery_array=finalQuery.split(" ")
# tf_query_matrix=[]
# for word in finalQuery_array:
#     word=word.lower()
#     totalfrequencyofword=re.findall(rf"\b{(word)}\b", finalQuery)
#     tf_query_matrix.append(len(totalfrequencyofword))
#
# tf_query_numpy_matrix=np.array(tf_query_matrix)
# print(tf_query_numpy_matrix)
# print(log_idf_ans)
# IDF_of_Query_vector=tf_query_numpy_matrix*log_idf_ans
# print(IDF_of_Query_vector)
#
#
# q = IDF_of_Query_vector.astype(float)
# D = Document_tf_idf_score_vector.astype(float)
# dot = D @ q ## matrix muplitpilcation with q as d[i].q @ means dot product in python
# D_norm = np.linalg.norm(D, axis=1)# mag of |d[i]|
# q_norm = np.linalg.norm(q) # mag of q |q|
#
# den = D_norm * q_norm # |d||q|
# cos_sims = np.divide(dot, den, out=np.zeros_like(dot), where=den!=0) # cos0=(dot/|d|*|q|)
# angles_deg = np.degrees(np.arccos(np.clip(cos_sims, -1.0, 1.0))) #0=cos^-1(dot/|d|*|q|)
#
# best_doc_index = np.argmax(cos_sims)
# print("best doc index:", best_doc_index, "best similarity:", cos_sims[best_doc_index])
# Indian_article_Data[Indian_article_Data.index==best_doc_index].values[0]
#
# ##LSA Starting.(Latent Semantics Analysis)
# '''
# now I want to minimize the corpus of document using SVD .
# Matrix Corpus of Document will be SVD into |D|=UΣV^T
# Σ= is the diagonal matrix of A^TA sq.root of eigenvalues first find the ATA
# '''
# # My A vector is document corpus .
# A=Document_tf_idf_score_vector
# print(type(A))# yes it's a numpy matrix
#
# A_transpose=np.transpose(A)
# print(A_transpose.shape)
# ## now A _tranpose * A
# ATA=A_transpose @ A
# ATA.shape
# #now i have ot find eigenvalues of ATA this is my square matrix so it will statify eq
# eigen_value=np.linalg.eigh(ATA)
# # now singluar values will be root of all eigen value
# singular_value=np.sqrt(eigen_value.eigenvalues)
# singular_value.sort()
# singular_value=singular_value[::-1]
# #therefore Σ or sigma =singular value diagonal matrix in descedning order.
# singular_value
# sigma_diagonal=np.diag(singular_value) # Σ or sigma
# sigma_diagonal
#
# #now we have to find U or VT
# # V is the eigenvector of ATA from the spectrum theorem
# V=eigen_value.eigenvectors
# print(V)
# V.shape # it's the orthogonal matrix.
# V_transpose=np.transpose(V)
# V_transpose
#
# ##Word2Vector Starting
# ##corpus of documents.
# # Cleaning the Documents removing the noise from it .
# def break_into_OneCorpus():
#     i=0
#     array_of_corpus_containing_sentences=[]
#     while(i<len(Indian_article_Data)):
#         data=Indian_article_Data.iloc[i]["articles"]
#         data=data.lower()
#         Split_Data=re.split(r'[,.!?]+', data)
#         array_of_corpus_containing_sentences.append(Split_Data)
#         i=i+1
#     return array_of_corpus_containing_sentences
# Sentences=break_into_OneCorpus()
# Sentences_Corpus=[]
# for sen in Sentences:
#     for each_sen in sen:
#         if(each_sen.strip() and len(each_sen.split(" "))>=6):
#             Sentences_Corpus.append(each_sen)
# len(Sentences_Corpus)
#
# Sentences_Corpus
#
# # We will train the first sentences in order to get embedding from it
#
# firstSen=Sentences_Corpus[0]
# cleanSentenceCorpus=[]
# for eachSen in Sentences_Corpus:
#     finalsen=eachSen.split(" ")
#     FinalWords=[]
#     for word in finalsen:
#        if "-" in word:
#           word=word.split("-")
#           FinalWords.extend(word)
#        else:
#         FinalWords.append(word)
#     finalSentence=""
#     for sen in FinalWords:
#        finalSentence=finalSentence+" "+sen
#        finalSentence.strip()
#     cleanSentenceCorpus.append(finalSentence)
#
# cleanSentenceCorpus # clean corpus with breaking hypen words in 2 words
#
#
#
# ## window size be 5
#  # absolute division //
#  # since 5 is the window size i am taking i will mid posiiton for first that is 5/2
#
# # We will train the first sentences in order to get embedding from it
# startpointer=0
# window_size=5
# endpointer=window_size
# cbow_dict=[]
# for sen in cleanSentenceCorpus:
#   startpointer=0
#   window_size=5
#   endpointer=window_size
#   FinalWords=sen.split()
#   while(endpointer<=len(FinalWords)):
#     i=startpointer
#     contextlist=[]
#     target=""
#     mid=(startpointer+endpointer )// 2
#     while(i<endpointer):
#         if(i!=mid):
#            context=FinalWords[i]
#            contextlist.append(context)
#         else:
#           target=FinalWords[i]
#         i=i+1
#     tuplecbow=(target,contextlist)
#     cbow_dict.append(tuplecbow)
#     startpointer=startpointer+1
#     endpointer=endpointer+1
#
# cbow_dict
#
# turple1=cbow_dict[10]
# mycontext=turple1[1]
# mycontext
# # hot en-coding of mycontext
# len(cleanSentenceCorpus)
# words=[]
# for sen in cleanSentenceCorpus:
#     words.extend(sen.split())
#
# #words
# #encoding
# ## now words is the collection of all the words but it should be unique so make it a list of set(words)
# word_voc=list(set(words))
# voc_len=len(word_voc)
# #customvect={}
# word_to_index={w:i for i,w in enumerate(word_voc)}
#
# def hotencode(word):
#     if word not in word_to_index:
#         print("word not in word_to_index")
#     vec=np.zeros(shape=(1,voc_len),dtype=int)
#     vec[0,word_to_index[word]]=1
#     return vec
#
# encodeword=hotencode("accident")
# print(encodeword)
#
# ##now will start training functions code
# embedding_dim=150
# w_input=hotencode("accident") # target
# making_hidden_matrix=np.random.uniform(low=-0.5, high=0.5, size=(voc_len,embedding_dim))
# h_result= w_input @ making_hidden_matrix # dimension
#
#
#
# ## output_weigth
# w_output=np.random.uniform(low=-0.5, high=0.5, size=(embedding_dim,voc_len))
# u_score=h_result @ w_output # shape would be 1 x voc_len
#
#
# # using softmax funciton to get the probability for it .
# def softmax(x):
#     exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return exp_x / np.sum(exp_x, axis=1, keepdims=True)
# ## softmax is essential as i need to get the probability bwt 0 & 1 so i know what's the probability of each word to be the context  since i need to compare it with encode which is in 0s and 1s [000010000] like this.
# y_pred=softmax(u_score)
#
# total_loss=0
# ## now entropy loss calculatiton.
# def calculateloss(word):
#     encode_word=hotencode(word)
#     y_true=np.array(encode_word)
#     loss = -np.sum(y_true * np.log(y_pred + 1e-9))
#     error_vector = y_pred - y_true               # (1 × voc_len)
#     return loss, y_true, y_pred, error_vector
#
#
# require_Data=cbow_dict[44]
#
# target=require_Data[0]# acciddent
# context=calculateloss(require_Data[1][0] )# into first word of context .
# print("my data ",require_Data)
# print("loss ",context[0])
# print("y_true",context[1])
# print("y_pred",context[2])
# print("error vector: ",context[3])
#
# #--backPropogation Starting.
# gradient=(h_result.transpose() ) @ context[3]
# # this gives me the gradient
#
# print("gradient shape ",gradient.shape)
# # now
# learning_rate=0.01 #(gives step by how much rate it should change the weight should not be too big or too short)
# #gradient_descend=w_output - (learning_rate*gradient) only for knoledge purpose
# w_output=w_output-(learning_rate*gradient)
#
# #Backprop to Hidden_layer_of_matrix.
# gradient_hidden_matrix=context[3] @ (w_output.transpose())
# target_index=word_to_index[target]
# making_hidden_matrix[target_index]=making_hidden_matrix[target_index]-(learning_rate * gradient_hidden_matrix) ## only the accident line to change .
# #---forward procession (changing everything with new weight values right )
#
#
# #forword propogation
# h_result=w_input @ making_hidden_matrix  # use old
# u_score=h_result @ w_output
# y_pred=softmax(u_score)
#
# context2=calculateloss(require_Data[1][0])
# print("loss 2 ",context2[0])
# print("y_true 2",context2[1])
# print("y_pred 2",context2[2])
# print("error vector 2: ",context2[3])
#

