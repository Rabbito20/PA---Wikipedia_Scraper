#Needed packages:
#install wikipedia
#install transliterate
#install pyOpenSSL --upgrade        #   This is not needed

import wikipedia
from multiprocessing import Pool
import transliterate as tr
import time
from functools import reduce
import collections
from bs4 import BeautifulSoup
import numpy as np

#   I deo - pristup tekstu za wikipedije
    #   1.  Dohvatamo stranice sa wikipedije
    #   2.  Koristimo Pool.map za paralelizaciju
    #   3.  Sanitizujemo trash podatke uz pomoc map/reduce-a
    #       Sanitizacija se može izvesti pokušajem da dohvatanja stranice na osnovu naslova (wikipedia.page(title)) i hvatanjem izuzetaka
    #   4.  Napisati map/reduce rešenje koje dohvata sažetke stranica (page.summary) iz sanitizovane liste formirane u prethodnoj tački.
    #   5.  Unifikujemo naslove (prevodimo sve naslove na cirilicno ili latinicno pismo iskljucivo)
    #   6.  Testiranje. Biramo nekoliko kljucnih reci, program vraca po 2 rezultata za svaku kljucnu rec.
####################################################################################################################

#   Ovo je zbog errora oko parsiranja kad scrapujemo wikipediju
#soup = BeautifulSoup(html, "html.parser")
#soup = BeautifulSoup(req1.text, 'html5lib')
#print(htmlDoc)
#lis = BeautifulSoup(html).find_all('li')
lang = 'sr'
wikipedia.set_lang(lang)
pool_num = 15

#   I deo
def get_pages(query, results=2):   #   2 rezultata
#def get_pages(query, results=50):    #   50 rezultata
    '''Dohvata naslove zahtevanog broja stranica koje se pojavaljuju kao 
    rezultati pretrage za zadatu kljucnu rec'''
    #transliterate.translit(pages, 'sr')        # idemo na stranicu i prolazimo kroz sve reci i prevodimo ih
    try:
        pages = wikipedia.search(query, results=results)
    except wikipedia.ERROR:
        print('ERROR - "{}" not found!'.format(query))

    return pages

#   Samo poredjenja radi
def get_pages_no_pool(stranice):
    start_time = time.time()
    for i in stranice:
        pages = get_pages(i)
        #print(pages)
        #print('----------------------------------------------------------------------------------------------\n')
    
    end_time = time.time() - start_time
    #print('NO POOL vreme je: ', end_time)
    print('\n------------------------------------------- NO POOL vreme je: %.3f' %end_time,' -------------------------------------------')

##########################################################
#   3.  Sanitizujemo trash podatke uz pomoc map/reduce-a

#   Krompiri su pomocne funkcije, nisam imao ideju za ime u trenutku pa sam ih tako nazvao
#   Krompir nam obavlja wikipedia.page()
def krompir(j):
    pov = []
    try:
        temp = wikipedia.page(title=j)
        pov.append(temp)
        #print(temp)
    except:
        #print('-----------  PAGE ERROR for "{}" -----------'.format(j))
        return ''
    return pov

#   Ovo nam je reducer
def redukovani_krompir(k1, k2):
    lista = []
    #rint(k1)
    #print(k2)
    #print()
    #for i in k1:
        #if i != None:
            #lista.append(i)
    #if k2 == None:
    for i in k2:
        if i != '' and (i in k1) != True:
            k1.append(i)
    
    return k1

#   mapa - mapirani podaci
def page_reduction(mapa):
    pool = Pool(pool_num)
    lista = []
    pov_lista = []
    
    #   Vrv postoji malo brzi nacin za ovo, ali zbog strukture i citljivosti je ovo bolje
    for i in mapa:
        #print('PR mapa:\n', i)
        res1 = pool.map(krompir, i)
        lista.append(res1)
        res2 = reduce(redukovani_krompir, lista)
        pov_lista.append(res2)

    pool.close()
    pool.join()
    #   Vracamo poslednji element samo,
    #   jer nam je poslednji element niz koji sadrzi skup svih Wiki stranica
    return pov_lista[-1]

##########################################################
#   1.  Dohvatamo stranice sa wikipedije
#   Samo mapiramo i vracamo listu svih rezultata
def get_pages_with_pool(stranice):
    start_time = time.time()
    pool = Pool(pool_num)
    #pages = wikipedia.search(query, results=results)

    queries = []
    for i in stranice:
        print('---------- {} '.format(i))
        queries.append(i)

    #   Mapirali smo query    
    result = pool.map(get_pages, queries)
    pool.close()    #   Zatvaramo pool kako nam se ne bi desio memory leak!
    pool.join()

    end_time = time.time() - start_time
    print('-------------------------------------------- MAPIRANJE vreme je: %.3f' %end_time, '--------------------------------------------')
    return result
##########################################################
#   4.  Page summary

    ######################
    #   el - trenutni
    #   el2 - prethodni /   el2[0] - recenica   /   el[1] - vrednost 0 ili 1
def extractor_reduce(el, el2):
    #pov0 = []
    pov = []
    #print(el2)
    if el2[1] == 1:
        #   Dodajemo el i el2
        #print('else Ubaci,', el, ' DODAJEMO ', el2[0])
        for i in el:
            if i == 1:
                continue
            pov.append(i)
        #pov.append(pov0)
        pov.append(el2[0])
        #return el[0], el2[0]
        #return pov
    else:
        #   Dodajemo samo el
        #print('Trash --- OSTAJE ISTO')
        for i in el:
            pov.append(i)
        #pov.append(pov0)
            #pov.append(el2[0])
            
    #print(pov)
    return pov
        #return el

    #extract_str = '\n        ⇌\n      \n    \n    {\\displaystyle \\rightleftharpoons }\n '
    #extract_str2 = '==\n\n\n='

    ######################

def extractor(element):
    #ret = None
    extract_str = '⇌'
    if str(element).__contains__('WikipediaPage'):
        temp1 = str(element).replace('<WikipediaPage ', '')
        temp = str(temp1).replace('>', '')
        try:
            #   Vraca sazetak teksta
            rt0 = wikipedia.summary(temp, sentences=5)    #   Drugi argument je broj stranica
            naslov = '\n' + str(temp) + '\n'
            rt = ''.join(rt0)
            rt = naslov + str(rt)
            ######################
            #   5.  Unifikujemo naslove
            #   Prevodi tekst u latinicu(reversed=True) ili cirilicu(reversed=False)
            ret = tr.translit(rt, lang, reversed=True)
            ######################
            #print('U extractoru')
            if ret.__contains__(extract_str):
                pom = ret
                ret.replace(pom, '')
                pov = ('', 0)
            else:
                #ret - tekst
                #temp - kljuc, naslov
                pov = (ret, 1)
                #print(pov)
                #print()
                #pov = (temp, 1)
            return pov
        except:
            #print('----------- SUMMARY ERROR for {} -----------\n'.format(temp))
            pov = ('', 0)
            return pov

def page_summary(lista):
    pool = Pool(pool_num)
    pov = []
    pool_res = pool.map(extractor, lista)
    pool.close()
    pool.join()

    pool_red = reduce(extractor_reduce, pool_res)

    #print('Duzina summary liste: ', len(pool_red))
    #print('##########################################################')
    #for i in pool_red:
        #tr.translit(i, 'sr', reversed=False)
        #print(i)
    #print('##########################################################')

    #print(pool_red)
    #ret = ''.join(pool_red)

    return pool_red

##########################################################
#   Ovo pozivamo u main-u
def fja1(stranice):
    start_time = time.time()
    mapp = get_pages_with_pool(stranice)
    
    pr = page_reduction(mapp)
    
    ps = page_summary(pr)
    for i in ps:
        print(i)
    #print('##########################################################')
    #for i in pr:
        #print('Stampamo PR: {}'. format(i))
        #print()
    #print('##########################################################')

    end_time = time.time() - start_time
    print('-------------------------------------------- vreme REDUKCIJE: %.3f' %end_time, '--------------------------------------------')

####################################################################################################################
####################################################################################################################
#   II deo - bag-of-words
#wikipedia.set_lang(lang)

#def get_pages(query, results=5):
    #pages = wikipedia.search(query, results=results)
    #return pages

##########################################################
#   Izdvajamo reci u recenici
def extract_words(sentence):
    words = []
    sentence = sentence.replace('–', ' ') 
    #print(sentence)
    for word in sentence.split():
        stripped_word = ''.join(ch for ch in word if ch.isalnum())
        #stripped_word = (stripped_word0.upper(), 1)
        #print(stripped_word)
        #   Stavljamo da nam sve reci budu velikim slovima
        words.append(stripped_word.upper())
    #print('extract_words')
    #print(words)
    return words

def dict_reduce(trenutni, sledeci):
    #print('Reduce')
    #print(trenutni)
    #print(sledeci)
    l = []
    if trenutni != None:
        for i in trenutni:
            if (i in l) == False and len(i) > 1:
                #print(i)
                l.append(i)
        for j in sledeci:
            #if len(j) <= 2:
                #print(j, ' -- duzina =', len(j))
            if (j in l) == False and len(j) > 1:
                #print(j)
                l.append(j)
    return l

#   Recenice razbija na reci i dodaje ih u 
def make_dictionary(sentences):
    pool = Pool(pool_num)
    words = []
    for sentence in sentences:
        #print('make_dictionary za:', sentence)
        w = pool.map(extract_words, sentences)
        #print(w, '\n')
        words.extend(w)         #   Dodajemo rec w na kraj liste
    #print(words)
    #   Redukovano
    red = reduce(dict_reduce, words)
    #print(red)

    #   No map_reduce
    #for sentence in sentences:
        #w = extract_words(sentence)
        #words.extend(w)         #   Dodajemo rec w na kraj liste
        #print(w)
    
    pool.close()
    pool.join()
    #words = sorted(list(set(words)))
    #print(words)
    return red

##########################################################
######################
#   No map reduce
def bag_of_words(sentence, words):
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(words), dtype=int)
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1
    print(sentence)
    print(bag)
    return np.array(bag)
######################
#   Reducer
def key_add(array, value):
    #print(array, '--------', value)
    if array and array[-1][0] == value[0]:
        array[-1] = array[-1][0], array[-1][1] + value[1]
    else:
        array.append(value)
    
    #print(array)
    #print()
    return array

#   Mapiramo reci
def map_words(sentence):
    words = []
    sentence = sentence.replace('–', ' ')
    
    #   Tuple rec i vrednost 1
    stripped_word = (''.join(ch for ch in sentence if ch.isalnum()), 1)
    #print(stripped_word)
    return stripped_word

#   With map reduce
def mapreduce_bag_of_words(sentence, words):
    pool = Pool(pool_num)
    #   Inicira vektore na 0
    bag = np.zeros(len(words), dtype=int)
    
    #   Lista reci iz recenice
    sentence_words = extract_words(sentence)
    
    #   No map
    #mapped_words = []
    #for sw in sentence_words:
      #mapped_words.append((sw,1))

    #   map_reduce
    mapped_words = pool.map(map_words, sentence_words)
    pool.close()
    pool.join()

    sorted_words = sorted(mapped_words, key = lambda x: x[0]) 
    #print(sorted_words)

    reduced = reduce(key_add, sorted_words, [])
    #print(reduced)
    #print('..............................................')

    for red in reduced:
        for i, cur in enumerate(words):
            if cur == red[0]:
                #print(i, '----', cur)
                bag[i] = red[1]
                #print(bag)
        #print()

    #print(sentence)
    #print(bag)
    #print()
    return np.array(bag)

##########################################################

def red_top10(t1, t2):
    pass
    

#   Trazi najcesce reci i najmanje koriscene reci
def common_words(text, dictionary, lower_bound, upper_bound):
    pool = Pool(pool_num)
    vectors = []
    for i in text:
        vectors.append(mapreduce_bag_of_words(i,dictionary))

    #print(vectors)
    #   Most common words / least occuring
    x = np.zeros(len(dictionary), dtype=int)
    #print(x)
    for v in vectors:
        x = x + v
    #print(vectors)
    
    mapped_top10 = []
    count10percent = 10               #int(len(vectors[0]) * 10/100)
    #print(count10percent)

    for v in vectors:
        temp = v.argsort()[-10:]  # trenutno 10, podesi da bude count10percent
        #print(v)
        for t in temp:
            mapped_top10.append((dictionary[t],1))

    #print(mapped_top10)
    sorted_top10_words = sorted(mapped_top10, key = lambda x: x[0]) 
    #print(sorted_top10_words)
    reduced_top10 = reduce(key_add, sorted_top10_words, [])
    #print('Reduced top 10\n', reduced_top10)
    #print()

    subst = []
    for w in reduced_top10:
        if(w[1] > upper_bound):
            subst.append(w)

    # manj od 1% ( ovde se radi sa manje od 2, zbog testiranja)
    #x = np.zeros(len(dictionary), dtype=int)
    #for v in vectors:
        #x = x + v
    #print(x)
    #print(np.sum(x))

    helper_vectors = []
    i = 0
    for ver in vectors:
        helper_vectors.append(np.where( vectors[i] > 0,1,0))
        i += 1
    
    sum = np.zeros(len(dictionary),dtype=int)
    for h in helper_vectors:
        #print(h)
        sum = sum + h
    #print(sum)

    ###################################
    lower_del = []
    nw =  np.where(sum < lower_bound)[0]
    #print('-->', nw)
    lower_del.append(nw)
    #print(lower_del)
    print('DUZINA TEKSTA:', len(text))
    print("Upper bound odsecanje :")
    print(subst)

    print("Lower bound odsecanje: ")

    for d in lower_del:
        i = 0
        for dd in d:
            print((dd,dictionary[dd-i]), end = '   ')
            dictionary.remove(dictionary[dd-i])
            i += 1


    for s in subst:
        dictionary.remove(s[0])
        #print(s[0])

    vectors2 = []
    #   Pravimo novi recnik
    for (i,d) in enumerate(dictionary):
        print(i,d, end = '  |   ')
    print('\n')

    for i in text:
        vectors2.append(mapreduce_bag_of_words(i,dictionary))

    #for v in vectors2:
        #print(v)
    
    return vectors2[-1]

##########################################################
def fja2(stranice):
    start_time = time.time()

    gp = get_pages_with_pool(stranice)
    pr = page_reduction(gp)
    tekst = page_summary(pr)
    #wikipedia.set_lang("en")
    #text_sample = wikipedia.summary("Serbia")
    #text_sample1 = wikipedia.summary("Serbian language")
    #text_sample2 = wikipedia.summary("Serb")
    #text_sample3 = wikipedia.summary("Belgrade")
    #tekst = [text_sample,text_sample1,text_sample2,text_sample3]

    #for p in tekst:
        #print(p)
    
    #   Kreiramo recnik
    dictionary = make_dictionary(tekst)
    #vectors = []

    #for (i,d) in enumerate(dictionary):
        #print(i, '-', d, end = '    \n')
    print()
    print('####################################################################################################################')

    #   Ubacujemo u vektore
    #for i in tekst:
        #bow = mapreduce_bag_of_words(i,dictionary)
        #vectors.append(bow)
        #print(bow)
        #print()

    #for i in vectors:
        #print(i)

    ######################
    # gornja i donja granica za brisanje
    lowBound = 1/100 * len(tekst)     # pojavljuje se u 1% 
    upBound = 90/100 * len(tekst)     # 90 %
    
    vectors2 = common_words(tekst, dictionary, lowBound, upBound)
    print('Finale fj2')
    #print(vectors2)
    #print('####################################################################################################################')
    ######################
    end_time = time.time() - start_time
    print('-------------------------------------------- vreme IZVRSENJA: %.3f' %end_time, '--------------------------------------------')
####################################################################################################################
####################################################################################################################
#   III deo - K-means klasterisanje
#   Treba da se zavrsi - ne radi
def fja3(stranice):
    '''
    K-means klastersanje putem MapReduce programa (5 bodova)
    Napisati map/reduce program koji vrši k-means klasterisanje liste vektora. 
    Nasumično generisati podatke iz nekoliko normalnih raspodela u dve dimenzije i pokazati da algoritam radi (videti vežbe za primer).
    '''

    k = 3

    n = len(vectors2[0])
    max_value = np.max(vectors2)

    centroids = np.random.uniform(0,max_value,(k,n))
    #print(centroids)

    cluster_assignments = np.zeros(len(vectors2), dtype=int)
    #print(cluster_assignments)
    for i,vek in enumerate(vectors2):
    print(i,vek)
    distance = np.sqrt(((centroids - vek)**2).sum(axis=1))
    cluster_assignments[i] = np.argmin(distance)

    print(cluster_assignments)

    for i,c in enumerate(centroids):
    cluster = [vek for j, vek in enumerate(vectors2) if cluster_assignments[j] == i]
    for temp in range(n):
        centroids[i][temp] = sum( x[temp] for x in cluster)/len(cluster)

    cluster_assignments = np.zeros(len(vectors2), dtype=int)
    print(cluster_assignments)
    for i,vek in enumerate(vectors2):
    print(i,vek)
    distance = np.sqrt(((centroids - vek)**2).sum(axis=1))
    cluster_assignments[i] = np.argmin(distance)

    print(cluster_assignments)
####################################################################################################################
#   IV deo - Primena resenja
#    Povezati kod iz zadataka 1-3 kojim se Wiki na srpskom jeziku pretražuje za sledeći skup ključnih reči 
#    ['Beograd', 'Prvi svetski rat', 'Protein', 'Mikroprocesor', 'Stefan Nemanja', 'Košarka'], 
#    za svaku dohvata po 50 rezultata, stranice prevode u bag-of-words reprezentaciju, i vrši njihovo klasterisanje.

#   main
if __name__ == "__main__":
    #stranice = ['Programiranje', 'Algoritmi', 'Strukture', 'Jahanje', 'Skijanja']
    #stranice = ['Beograd', 'Prvi svetski rat', 'Protein', 'Mikroprocesor', 'Stefan Nemanja', 'Košarka']
    stranice = ['Beograd', 'Jezik', 'Protein', 'Srbija', 'Stefan Nemanja']
    
    #   I deo
    #   Za svaku kljucnu rec vraca po nekoliko (2) rezultata
    fja1(stranice)

    #   II deo
    #   Bag of words
    #titles = ['Pera voli da gleda filmove. Mika takođe voli filmove.' , 'Mika voli da gleda i fudbalske utakmice.' , 'Mika Mika Mika voli da gleda filmove. Pera takođe.']
    #fja2(stranice)

    print('\n-- Kraj --\n')