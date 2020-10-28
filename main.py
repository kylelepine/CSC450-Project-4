import unittest
# from fallDetection import dectectFall

#Basic Framework For Unit Testing Fall Detection

class TestFallCases(unittest.TestCase):
    
    def test0_5(self):
        #Video file for 0-5 feet
        testFile = True
        self.assertEqual(testFile,True)
    
    def test5_10(self):
        #Video file for 5-10 feet
        testFile = True
        #run method for fall detection using test video. Return boolean
        self.assertEqual(testFile,True)
    
    def test10_15(self):
        #Video file for 10-15 feet
        testFile = True
        self.assertEqual(testFile,True)
    
    def test15_20(self):
        #Video file for 15-20 feet
        testFile = True
        #run method for fall detection using test video. Return boolean
        self.assertEqual(testFile,True)
    
    def test20_25(self):
        #Video file for 20-25 feet
        testFile = True
        #run method for fall detection using test video. Return boolean
        self.assertEqual(testFile,True)

    def testLowLight(self):
        #Video file for low light
        testFile = True
        #run method for fall detection using test video. Return boolean
        self.assertEqual(testFile,True)
    
    def testObstructed(self):
        #Video file for obstructed view
        testFile = True
        #run method for fall detection using test video. Return boolean
        self.assertEqual(testFile,True)
    
    


if __name__ == '__main__':
    unittest.main()