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
"""
    def test_getFormattedData(self):
        document = [
            {
                "docname": "test",
                "doc": 
                [
                    {
                        "text": "The",
                        "break_level": "NO_BREAK"
                    },
                    {
                        "text": "man",
                        "break_level": "SPACE_BREAK"
                    },
                    {
                        "text": "jumped",
                        "lemma": "jump",
                        "break_level": "SPACE_BREAK",
                        "sense": "jump_sense1"
                    },
                    {
                        "text": "at",
                        "break_level": "SPACE_BREAK"
                    },
                    {
                        "text": "the",
                        "break_level": "SPACE_BREAK"
                    },
                    {
                        "text": "store",
                        "break_level": "SPACE_BREAK",
                        "sense": "store_sense2"
                    },
                    {
                        "text": ".",
                        "break_level": "NO_BREAK"
                    },
                    """ {
                        "text": "The",
                        "break_level": "SENTENCE_BREAK"
                    },
                    {
                        "text": "man",
                        "break_level": "SPACE_BREAK"
                    },
                    {
                        "text": "jumped",
                        "lemma": "jump",
                        "break_level": "SPACE_BREAK",
                        "sense": "jump_sense1"
                    },
                    {
                        "text": "again",
                        "break_level": "SPACE_BREAK"
                    },
                    {
                        "text": ".",
                        "break_level": "NO_BREAK"
                    } """
                ]
            }
            correct_output = [
                {
                    "docname": "test",
                    "doc": [
                        {
                            "natural_sent": "The man jumped at the store.",
                            "sent": [
                                "The",
                                "man",
                                "jumped",
                                "at",
                                "the",
                                "store",
                                "."
                            ],
                            "bert_sent": [
                                "The",
                                "man",
                                "jumped",
                                "at",
                                "the",
                                "store",
                                "."
                            ],
                            "tracking": [
                                0,
                                1,
                                2,
                                3,
                                4,
                                5,
                                6
                            ],
                            "senses": [
                                {
                                "word": "jumped",
                                "pos": 2,
                                "sense": "jump_sense1"
                                },
                                {
                                "word": "store",
                                "pos": 5,
                                "sense": "store_sense2"
                                }
                        ]
                    }
                ]
            }
        ]
        assert(getFormattedData("test", document) == correct_output)
"""

if __name__ == "__main__":
	unittest.main()
