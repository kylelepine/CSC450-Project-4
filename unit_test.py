
import cv2
import unittest
import ComputerVision
import Templates
import HumanStateClassifier

class TestFallCasesHelper:
    def displayTestCV(self, foregroundClassifier, edgeClassifier, fileName):
        # Computer Vision
        return ComputerVision.display(foregroundClassifier=foregroundClassifier,
                                    edgeClassifier=edgeClassifier,
                                    videoPath=fileName,
                                    saveTemplate=False,
                                    checkTemplate=True)

#Basic Framework For Unit Testing Fall Detection

class TestFallCases(unittest.TestCase):
    # Load Templates Locally
    def test0_5(self):
        # Load Templates
        templates = Templates.loadTemplatesLocally()

        # Classifiers
        test_case_helper = TestFallCasesHelper()
        edge_classifier = HumanStateClassifier.KNeighborsClassifier(templates['edge'], k=4)
        foreground_classifier = HumanStateClassifier.KNeighborsClassifier(templates['foreground'], k=4)

        # Video file for 0-5 feet
        fileName = './fall_samples/fall-01-cam0.mp4'
    
        fall_detected = test_case_helper.displayTestCV(foregroundClassifier=foreground_classifier, edgeClassifier=edge_classifier, fileName=fileName)

        self.assertEqual(fall_detected, True)

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
    
    def testDatabaseConnection(self):
        # Test Connection With Database
        # Query Data or get status
        test = True

if __name__ == '__main__':
    unittest.main()