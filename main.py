# Modules 
import UserInterface
import TemplateViewer
import Templates
import HumanStateClassifier
import ComputerVision

def main():
    print('Starting FDSystem')

    # Database credentials. 
    LOCAL_DATABASE_NAME = 'CSC-450_FDS'
    LOCAL_DATABASE_PASSWORD = 'Pd9;$chsi9-$nc'

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
    else:
        templates = Templates.loadTemplatesLocally()

    edge_classifier = HumanStateClassifier.KNeighborsClassifier(templates['edge'], k=20)
    foreground_classifier = HumanStateClassifier.KNeighborsClassifier(templates["foreground"], k=20)

    UserInterface.systemUserInterface(templates,
                                    database,
                                    edgeClassifier=edge_classifier,
                                    foregroundClassifier=foreground_classifier)

if __name__ == '__main__':
    main()