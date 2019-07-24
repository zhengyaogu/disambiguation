import unittest
from compare import *

class TestCompare(unittest.TestCase):
	
	def test_track(self):
		sent = ["But", "there", "\'", "##s", "no", "rea", "##son"]
		tracking = TrackRawSentenceIndices(sent)
		assert tracking == [0, 1, 2, 2, 3, 4, 4]

	def test_raw_to_bert(self):
		sent = ["I", "am", "gobbling"]
		bert_sent = getBertSentenceFromRaw(sent)
		assert bert_sent == ["i", "am", "go", "##bbling"]



if __name__ == "__main__":
	unittest.main()
