from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sent_mod as s 

#consumer key, consumer secret, access token, access secret.
ckey="RbPHdBQPf3uzrv8giPUiZIsrW"
csecret="GvTqlSu2rbvhH7vXqdt2s5IGny8LycZ0IC9uw8JAAbWpdc4MX2"
atoken="1031145620352974850-CBXW51vYN5OeNtyTe3OlPMihS1H5UB"
asecret="w8Td4pE58x8VL8By11XBfMOrKiOuB7m1js46sXPiRqA3i"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence*100 >= 80:
        	output = open('twitter-out.txt', 'a')
        	output.write(sentiment_value)
        	output.write('\n')
        	output.close()
        
        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["obama"])