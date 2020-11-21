
import cv2
import unittest
import psycopg2
import ComputerVision
import Templates
import HumanStateClassifier

class TestFallCasesHelper:
    def displayTestCV(self, local, fileName):
        templates = self.loadTemplates(local=local)

        # Classifiers
        edge_classifier = HumanStateClassifier.KNeighborsClassifier(templates['edge'], k=4)
        foreground_classifier = HumanStateClassifier.KNeighborsClassifier(templates['foreground'], k=4)

        # Computer Vision
        return ComputerVision.display(foregroundClassifier=foreground_classifier,
                                    edgeClassifier=edge_classifier,
                                    videoPath=fileName,
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
    def test0_5(self):
        # Video file for 0-5 feet
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-01-cam0.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def test5_10(self):
        # Video file for 5-10 feet
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-02-cam0.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)
    
    def test10_15(self):
        #Video file for 10-15 feet
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-03-cam0.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)
    
    def test15_20(self):
        #Video file for 15-20 feet
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-04-cam0.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)
    
    def test20_25(self):
        #Video file for 20-25 feet
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-05-cam0.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def testLowLight(self):
        #Video file for low light
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-06-cam0.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)
    
    def testObstructed(self):
        #Video file for obstructed view
        test_case_helper = TestFallCasesHelper()
        fileName = './fall_samples/fall-07-cam0.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
        self.assertEqual(fall_detected, True)

    def testCamera(self):
        test = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            test = False
        self.assertEqual(test,True)
    
    def testDBConnection(self):
        try:
            self.conn = psycopg2.connect(host = 'localhost', \
                database = 'CSC-450_FDS', user = 'postgres', \
                password = 'Apcid28;6jdn')
            test = True
        except (Exception, psycopg2.DatabaseError):
            test = False
        self.assertEqual(test, True)

if __name__ == '__main__':
    unittest.main()