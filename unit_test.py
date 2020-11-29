
import cv2
import unittest
import psycopg2
import ComputerVision
import Templates
import HumanStateClassifier

class TestFallCasesHelper:
    def displayTestCV(self, local, fileName, displayForeground=False):
        templates = self.loadTemplates(local=local)

        # Classifiers
        edge_classifier = HumanStateClassifier.KNeighborsClassifier(templates['edge'], k=4)
        foreground_classifier = HumanStateClassifier.KNeighborsClassifier(templates['foreground'], k=4)

        # Computer Vision
        return ComputerVision.display(foregroundClassifier=foreground_classifier,
                                    edgeClassifier=edge_classifier,
                                    videoPath=fileName,
                                    displayForeground=displayForeground,
                                    saveTemplate=False,
                                    checkTemplate=True)

    def loadTemplates(self, local):
        if (local == True):
            # Load Templates Locally
            templates = Templates.loadTemplatesLocally()
        else:
            # Load Templates from Database
            LOCAL_DATABASE_NAME = "postgres"
            LOCAL_DATABASE_PASSWORD = "password"
            database = Templates.TemplateDatabase(LOCAL_DATABASE_NAME, LOCAL_DATABASE_PASSWORD)
            database.connect()
            templates = {'edge': {}, 'foreground': {}}
            template_characteristics = templates.keys()
            template_types = ['upright', 'falling', 'sitting', 'lying']

            for template_characteristic in template_characteristics:
                for template_type in template_types:
                    templates[template_characteristic][template_type] = []
            if database.connected():
                templates = database.load_templates(templates)

        return templates

# Basic Framework For Unit Testing Fall Detection

class TestFallCases(unittest.TestCase):
    def test0_5_female(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-0-5-1.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def test0_5_foreward_fall(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-0-5-2.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)
    
    def test0_5_backward_fall(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-0-5-3.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)
    
    def test5_10_forewards(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-5-10-1.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def test5_10_backwards(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-5-10-2.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)
    
    def test15_20(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-15-20-1.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def test15_20(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-15-20-2.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def testLowLight_15_20(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-lowlight.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)
    
    def testObstructed_15_20(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-obstructed.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def testDetectHuman(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/dogs.mp4'
        fall_detected = False
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, False)

    def testSittingDown(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/human-sitting-down.mp4'
        fall_detected = False
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, False)

    def testLyingDown(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/human-lying-down.mp4'
        fall_detected = False
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, False)

    def testRemoveNonMovingEntites(self):
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/static-with-window.mp4'
        test_case_helper.displayTestCV(local=True, fileName=fileName, displayForeground=True)

    def testCamera(self):
        test = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            test = False
        self.assertEqual(test,True)
    
    def testDBConnection(self):
        test_case_helper = TestFallCasesHelper()
        db_templates = test_case_helper.loadTemplates(local=False)
        self.assertGreater(len(db_templates), 0)

if __name__ == '__main__':
    unittest.main()