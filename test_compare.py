import unittest
from compare import *

class TestCompare(unittest.TestCase):
	
	def test_track(self):
		raw_sent = ["/", "written", "/", "letters", "/", "112C","-", "L014.txt", "June", "26", ",", "1995", "Dear", "Friend", ":"]
		bert_sent = ["/", "written", "/", "letters", "/", "112", "##c", "-", "l", "##01", "##4", ".", "tx", "##t", "june", "26", ",", "1995", "dear", "friend", ":"]
		tracking = trackRawSentenceIndices(raw_sent, bert_sent)
		print(tracking)
		assert tracking == [0, 1, 2, 3, 4, 5, 5, 6, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14]

	def test_raw_to_bert(self):
		sent = ["I", "am", "gobbling"]
		bert_sent = getBertSentenceFromRaw(sent)
		assert bert_sent == ["i", "am", "go", "##bbling"]



if __name__ == "__main__":
	unittest.main()
