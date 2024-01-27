def reverse_string(s):
    """
    This function takes a string as input and returns the string reversed.
    """
    return s[::-1]

import unittest

class TestReverseString(unittest.TestCase):

    def test_reverse_string(self):
        s = "Hello, World!"
        self.assertEqual(reverse_string(s), "dlroW ,olleH")

    def test_reverse_string_empty(self):
        self.assertEqual(reverse_string(""), "")

    def test_reverse_string_non_string(self):
        self.assertEqual(reverse_string(123), "")

if __name__ == '__main__':
    unittest.main()