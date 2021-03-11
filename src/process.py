
def processfile (filename,outfilename):
    stopwords = set()
    with open('.././data/stopwords_1.txt', 'r') as f:
        for stopword in f.readlines():
            stopword = stopword.rstrip("\n")
            stopwords.add(stopword)

    with open(filename, 'r', encoding='utf-8') as f:
        outstr = ""
        for line in f:
            line = line.strip().split()
            for i in range (len(line)):
                if line[i] not in stopwords:
                    outstr += line[i] + " "
            outstr += "\n"
    outputs = open(outfilename,'w', encoding="utf8")
    outputs.write(outstr)





