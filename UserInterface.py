import ComputerVision
import TemplateModifier
import Templates



def systemUserInterface(templates, database, foregroundClassifier, edgeClassifier):

    while True:

        print("""
        Command:(button)              Description:
        view_video:(1)                Displays available videos in 'fall_samples' with computer vision.
        view_webcam:(2)               Displays connected webcam with computer vision
        modify templates:(3)          Allows user to modify templates that exist in the database.
        compare_template:(4)          Demonstrates comparing a template to a frame.
        database:(5)                  Access Database UI.
        load templates from file:(6)  Loads templates from files manually instead of the database. 
        quit:(q)
        """)

        command = input("Command: ")

        if command == '1':

            available_videos = ['./fall_samples/fall-01-cam0.mp4', './fall_samples/fall-27-cam0.mp4']

            print("Please choose from: ")

            i = 0
            for video in available_videos:
                print(str(i) + video)
                i+=1

            try:
                selection = int(input("Enter: "))

                if selection < i:

                    print("Would you like to save the frames as templates?(y/n):")
                    save_templates = input()
                    save_templates = True if save_templates == 'y' else False

                    print("Would you like to compare templates to video frame?(y/n):")
                    check_templates = input()
                    check_templates = True if check_templates == 'y' else False

                    ComputerVision.display(foregroundClassifier=foregroundClassifier,
                                        edgeClassifier=edgeClassifier,
                                        videoPath = available_videos[selection], 
                                        saveTemplate = save_templates, 
                                        checkTemplate = check_templates)
                else:

                    print("Incorrect selection.")

            except ValueError as error:

                print(error)

        elif command == '2':

            session_name = input("Session Name (leave blank for default session name): ")

            if session_name == "":

                ComputerVision.display(foregroundClassifier=foregroundClassifier,
                                    edgeClassifier=edgeClassifier,
                                    saveTemplate=False,
                                    checkTemplate=False)
            else:

                ComputerVision.display(foregroundClassifier=foregroundClassifier,
                                    edgeClassifier=edgeClassifier,
                                    saveTemplate=True,
                                     checkTemplate=False,
                                      sessionName= session_name)
                                      
        elif command == '3':

            template_modifier = TemplateModifier.TemplateModifier(templates)
            template_modifier.crop_template()

        elif command == '4': 
            
            comparison_frame = templates['edge']['upright'][0]
            ComputerVision.showImage(comparison_frame)

            edge_classification = edgeClassifier.classify(comparison_frame)
            print(f'edge_classification: {edge_classification}')

            foreground_classification = foregroundClassifier.classify(comparison_frame)
            print(f'foreground_classification: {foreground_classification}')

        elif command == '5':

            if database.connected():
                databaseUserInterface(database)
                templates = database.loadTemplates()

            else:
                print("Database not connected.")

        elif command == '6':
            templates = Templates.loadTemplatesLocally()

        elif command == 'q':
            break

        else:
            print("incorrect command.")

def databaseUserInterface(database):

    database.connect()

    show_UI = database.connected()

    while show_UI:

        print("""
        Command:                        Description:\n
        1                               Add template.\n
        2                               Delete template.\n
        3                               Access template image by template_id.\n
        4                               Upload all templates locally.\n
        5                               Delete all entries.\n
        Previous Menu(r)                Returns to preivous menu.
        """)

        command = input("Enter a command: ")
        
        if command == "1":

            image_path = input("Please enter the path of the image: ")
            template_type = input("Please enter the template_type: ")
            template_characteristic = input("Please enter template_characteristic: ")
            
            image_byte_array = ComputerVision.imagePathToByteArray(image_path)

            image_name = image_path.split('/')
            image_name = image_name[-1]

            database.add_template(template_type, template_characteristic, image_name, image_byte_array)

        elif command == "2":

            template_id = input("Enter the template_id you wish to delete: ")
            database.delete_template(template_id)

        elif command == "3":

            template_id = input("Enter template_id: ")
            image = database.access_image_by_id(template_id)
            ComputerVision.showImage(image)

        elif command == "4":

            database.upload_all_local_templates()
        
        elif command == "5":

            id_list = database.list_of_all_IDs()

            for template_id in id_list:
                database.delete_template(template_id)

        elif command == "r":

            show_UI = False

        else:
            print("\nINCORRECT COMMAND ENTERED")
