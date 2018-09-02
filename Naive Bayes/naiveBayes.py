# tokenizing - word; sentence;
# lexicon and corporas
# corpora - body of text about a single or related topics. 
#           Eg- presidential speeches, English text.
# lexicon - word and their meanings; meanings can be situational.
#           invester-speak .... regualar-english-speak

# Stop Words - words that aren't useful in data analysis. Eg- "himself"

# stemming -  normalization of words, reducing redundancy.
# 			  Eg- "I was taking a ride in the car."
#                 "I was riding in the car."

import nltk
import random, math, pickle
from nltk.corpus import stopwords, state_union, gutenberg, wordnet, movie_reviews
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

####  Tokenizing and filtering stop words ####

# example_txt = 'This is an example showing off word filteration.'
# stop_words = set(stopwords.words("english"))

# words = word_tokenize(example_txt)
# filtered_words = [w for w in words if w not in stop_words]
# print(filtered_words)

#### Stemming ####

# ps = PorterStemmer()
# example_words = ['python', 'pythoner', 'pythoning', 'pythoned', 'pythonly']
# example_txt = "A good pythoner is always pythoning pythonly with python, even though every pythoner has pythoned poorly at least once in the past."

# words = word_tokenize(example_txt)
# for w in words:
	# print(ps.stem(w))

#### Part of Speech Tagging ####

# # POS tag list:

# # CC	coordinating conjunction
# # CD	cardinal digit
# # DT	determiner
# # EX	existential there (like: "there is" ... think of it like "there exists")
# # FW	foreign word
# # IN	preposition/subordinating conjunction
# # JJ	adjective	'big'
# # JJR	adjective, comparative	'bigger'
# # JJS	adjective, superlative	'biggest'
# # LS	list marker	1)
# # MD	modal	could, will
# # NN	noun, singular 'desk'
# # NNS	noun plural	'desks'
# # NNP	proper noun, singular	'Pranjal'
# # NNPS	proper noun, plural	'Americans'
# # PDT	predeterminer	'all the kids'
# # POS	possessive ending	parent's
# # PRP	personal pronoun	I, he, she
# # PRP$	possessive pronoun	my, his, hers
# # RB	adverb	very, silently,
# # RBR	adverb, comparative	better
# # RBS	adverb, superlative	best
# # RP	particle	give up
# # TO	to	go 'to' the store.
# # UH	interjection	errrrrrrrm
# # VB	verb, base form	take
# # VBD	verb, past tense	took
# # VBG	verb, gerund/present participle	taking
# # VBN	verb, past participle	taken
# # VBP	verb, sing. present, non-3d	take
# # VBZ	verb, 3rd person sing. present	takes
# # WDT	wh-determiner	which
# # WP	wh-pronoun	who, what
# # WP$	possessive wh-pronoun	whose
# # WRB	wh-abverb	where, when

# train_txt = state_union.raw('2005-GWBush.txt')
# sample_txt = state_union.raw('2006-GWBush.txt')
# custom_sent_tokenizer = PunktSentenceTokenizer(train_txt)
# tokenized = custom_sent_tokenizer.tokenize(sample_txt)

# def process_content():
# 	try:
# 		for sent in tokenized:
# 			words = word_tokenize(sent)
# 			tagged = nltk.pos_tag(words)
# 			print(tagged)

# 	except Exception as e:
# 		print(str(e))

# process_content()

#### Chunking and Chinking ####

# train_txt = state_union.raw('2005-GWBush.txt')
# sample_txt = state_union.raw('2006-GWBush.txt')
# custom_sent_tokenizer = PunktSentenceTokenizer(train_txt)
# tokenized = custom_sent_tokenizer.tokenize(sample_txt)

# def process_content():
# 	try:
# 		for sent in tokenized:
# 			words = word_tokenize(sent)
# 			tagged = nltk.pos_tag(words)
			
# 			# chunkGram = r'''Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}'''
# 			chunkGram = r'''Chunk: {<.*>+}
# 									}<VB.?|IN|DT>+{'''
# 			chunkParser = nltk.RegexpParser(chunkGram)
# 			chunked = chunkParser.parse(tagged)

# 			chunked.draw()

# 	except Exception as e:
# 		print(str(e))

# process_content()

#### Named Entity ####

# # NE Type	Examples:

# # ORGANIZATION	Georgia-Pacific Corp., WHO
# # PERSON	Eddy Bonte, President Obama
# # LOCATION	Murray River, Mount Everest
# # DATE	June, 2008-06-29
# # TIME	two fifty a m, 1:30 p.m.
# # MONEY	175 million Canadian Dollars, GBP 10.40
# # PERCENT	twenty pct, 18.75 %
# # FACILITY	Washington Monument, Stonehenge
# # GPE	South East Asia, Midlothian

# train_txt = state_union.raw('2005-GWBush.txt')
# sample_txt = state_union.raw('2006-GWBush.txt')
# custom_sent_tokenizer = PunktSentenceTokenizer(train_txt)
# tokenized = custom_sent_tokenizer.tokenize(sample_txt)

# def process_content():
# 	try:
# 		for sent in tokenized:
# 			words = word_tokenize(sent)
# 			tagged = nltk.pos_tag(words)

# 			namedEnt = nltk.ne_chunk(tagged, binary=True)
# 			namedEnt.draw()

# 	except Exception as e:
# 		print(str(e))

# process_content()

#### Lemmatizing: Stemming but the output is an actual word ####

# lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize('cats'))
# print(lemmatizer.lemmatize('cacti'))
# print(lemmatizer.lemmatize('geese'))
# print(lemmatizer.lemmatize('rocks'))
# print(lemmatizer.lemmatize('python'))

# # pos is by default set to 'n' (noun), so if a non-noun is passed
# # it's pos must be specified
# print(lemmatizer.lemmatize('better', pos='a'))
# print(lemmatizer.lemmatize('best', pos='a'))
# print(lemmatizer.lemmatize('run'))
# print(lemmatizer.lemmatize('run', pos='v'))

#### WordNet ####

# syns = wordnet.synsets('program')

# # synset
# print(syns)
# print(syns[0].name())

# # just the word
# print(syns[0].lemmas()[0].name())

# # definition
# print(syns[0].definition())

# # examples
# print(syns[0].examples())

# syns = wordnet.synsets('good')

# synonyms, antonyms = [], []
# for syn in syns:
# 	for l in syn.lemmas():
# 		synonyms.append(l.name())
# 		if l.antonyms():
# 			antonyms.append(l.antonyms()[0].name())

# print(set(synonyms))
# print(set(antonyms))

# w1 = wordnet.synset('ship.n.01')
# w2 = wordnet.synset('boat.n.01')

# # WuPalmer Algorithm: range[0, 1]
# # wp = (2 X depth(lcs)) / (depth(synset1) + depth(synset2))

# # Least Common Subsumer (LCS) of two concepts A and B is 
# # "the most specific concept which is an ancestor of both A and B",
# # where the concept tree is defined by the is-a relation.

# # Depth: the distance to their top most hypernym.
# # Hypernym: a word with a broad meaning.
# print(w1.wup_similarity(w2))

# w1 = wordnet.synset('ship.n.01')
# w2 = wordnet.synset('car.n.01')
# print(w1.wup_similarity(w2))

# w1 = wordnet.synset('ship.n.01')
# w2 = wordnet.synset('cactus.n.01')
# print(w1.wup_similarity(w2))

#### Text Classification ####

documents = [(list(movie_reviews.words(fileid)), category)
			for category in movie_reviews.categories()
			for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
# print(documents[1])

all_words = []
for word in movie_reviews.words():
	all_words.append(word.lower()) # normalize to lower

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words['stupid'])

word_features = list(all_words.keys())[:3000]

def find_features(document):
	words = set(document) # remove repetition
	features = {}
	for word, i in zip(word_features, range(0, 3000)):
		features[word] = ((word in words), i/(3000))

	return features

# print(find_features(documents[0][0]))
featureSets = [(find_features(rev), category)
				for (rev, category) in documents]

trainingSet = featureSets[:1900]
testingSet = featureSets[1900:]

# Naive Bayes Algorithm:
# posterior = (prior occurance x likelihood) / evidence
# P(A/B) = P(B/A)*P(A) / P(B)
# Basic algo, requires less computation, so it's easy to understand 
# and highly scalable

# classifier = nltk.NaiveBayesClassifier.train(trainingSet)
# print('Naive Bayes Accuracy (%):', (nltk.classify.accuracy(classifier, testingSet))*100)
# classifier.show_most_informative_features(15)

# save_classifier = open('sentiment.pickle', 'wb')
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

# classifier_f = open('sentiment.pickle', 'rb')
# classifier = pickle.load(classifier_f)
# classifier_f.close()

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector[0])
	return separated
 
def mean(wordDict):
	numbers = []
	for word in wordDict:
		try:
			if wordDict[word][0]:
				numbers.append(wordDict[word][1])
			else:
				numbers.append(-1*wordDict[word][1])
		except:
			continue

	try:
		return sum(numbers)/float(len(numbers))
	except:
		return 0
 
def stdev(wordDict):
	numbers = []
	for word in wordDict:
		try:
			if wordDict[word][0]:
				numbers.append(wordDict[word][1])
			else:
				numbers.append(-1*wordDict[word][1])
		except:
			continue

	try:
		avg = mean(numbers)
		variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
		return math.sqrt(variance)
	except:
		return 0
 
def summarize(dataset):
	summaries = [(mean(attributeDict), stdev(attributeDict)) for attributeDict in dataset]
	# del summaries[-1]
	return summaries
 
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries
 
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[0]

			numbers = [0]
			for word in x:
				try:
					if x[word][0]:
						numbers.append(x[word][1])
					else:
						numbers.append(-1*x[word][1])
				except:
					continue

			for number in numbers:
				probabilities[classValue] *= calculateProbability(number, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

# print(trainingSet[0])
summaries = summarizeByClass(trainingSet)
predictions = getPredictions(summaries, testingSet)
accuracy = getAccuracy(testingSet, predictions)
print('Accuracy:', accuracy)
