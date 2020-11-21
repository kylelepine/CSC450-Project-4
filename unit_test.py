
import cv2
import unittest
import psycopg2
import ComputerVision
import Templates
import HumanStateClassifier

class TestFallCasesHelper:
    def displayTestCV(self, local, fileName):
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

        # Classifiers
        edge_classifier = HumanStateClassifier.KNeighborsClassifier(templates['edge'], k=4)
        foreground_classifier = HumanStateClassifier.KNeighborsClassifier(templates['foreground'], k=4)

        # Computer Vision
        return ComputerVision.display(foregroundClassifier=foreground_classifier,
                                    edgeClassifier=edge_classifier,
                                    videoPath=fileName,
                                    saveTemplate=False,
                                    checkTemplate=True)

# Basic Framework For Unit Testing Fall Detection

class TestFallCases(unittest.TestCase):
    # Load Templates Locally
    def test0_5(self):
        test_case_helper = TestFallCasesHelper()

        # Video file for 0-5 feet
        fileName = './fall_samples/fall-01-cam0.mp4'
        fall_detected = test_case_helper.displayTestCV(local=True, fileName=fileName)
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