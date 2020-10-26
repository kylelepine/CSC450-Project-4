import psycopg2
import numpy as np
from cv2 import cv2
from os import walk

# Database class to handle pgfunctionality
class FDSDatabase:
    conn = None
    
    template_types = ['upright', 'falling', 'sitting', 'lying']
    template_characteristics = ['edge', 'foreground']
    template_dictionary = {'edge': {}, 'foreground': {}}

    def __init__(self, databaseName, databasePassword):
        self.database_name = databaseName
        self.database_password = databasePassword
    
    # FDSDatabase methods
    def connected(self):
        if self.conn is not None:
            return True
        else:
            return False

    def connect(self):
        try:
            # Attempts to connect to server
            print("Connecting...")
            self.conn = psycopg2.connect(host = 'localhost', \
                database = self.database_name, user = 'postgres', \
                password = self.database_password)
            print("Connection successful.")
            self.print_db_version()  
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
      
    def disconnect(self):
        # closes communcation with PostgreSQL server
        print("Disconnecting...")
        if self.conn is not None:
            self.conn.close()
        print("Disconnection successful.")

    def print_db_version(self):
        print('PostgreSQL database version:')
        try:
            curr = self.conn.cursor()
            curr.execute('SELECT version()')
            db_version = curr.fetchone()
            curr.close()
            print(db_version)
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def add_template(self, templateType, templateCharateristic, imageName, imageByteArray):
        try:
            curr = self.conn.cursor()
            print("adding template...")
            curr.execute('''
            INSERT INTO template (template_type, template_characteristic, image_name, image)
            VALUES(%s, %s, %s, %s)''', (templateType, templateCharateristic, imageName, imageByteArray))
            print("Updating template table...")
            self.conn.commit()
            curr.close()
            print("Added template successfully.")
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        curr.close()
        
    def delete_template(self, templateId):
        try:
            curr = self.conn.cursor()
            curr.execute('''
            DELETE FROM template
            WHERE template_id = %s
            ''', (templateId,))
            self.conn.commit()
            curr.close()
            print("Deleted template successfully.")
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def access_image_by_id(self, templateId):
        try:
            curr = self.conn.cursor()
            curr.execute('''
            SELECT image
            FROM template
            WHERE template_id = %s
            ''', (templateId,))
            template_bytes = curr.fetchone()
            template = byteStringToImage(template_bytes[0].tobytes())
            curr.close()
            return template
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        
    def access_all_image_by_type_and_chr(self, templateType, templateCharacteristic):
        try:
            curr = self.conn.cursor()
            curr.execute('''
            SELECT template_id, image
            FROM template
            WHERE (template_type = %s) AND (template_characteristic = %s)
            ''', (templateType, templateCharacteristic))
            rows = curr.fetchall()
            template_type_array = []
            for row in rows:
                template_bytes = row[1].tobytes()
                template = byteStringToImage(template_bytes)
                template_type_array.append(template)
                
            curr.close()
            print(f"Successfully loaded {templateCharacteristic} {templateType} array.")
            return template_type_array
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
    
    def list_of_all_IDs(self):
        try:
            curr = self.conn.cursor()
            curr.execute('''
            SELECT template_id
            FROM template
            ''', )
            rows = curr.fetchall()
            # print(rows)
            id_list = []
            for row in rows:
                id_list.append(str(row[0]))
            curr.close()
            return id_list
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def load_template_dictionary(self):
        for template_type in self.template_types:
            for characteristic in self.template_characteristics:
                self.template_dictionary[characteristic][template_type] = \
                    self.access_all_image_by_type_and_chr(template_type, characteristic)
        return self.template_dictionary

def userInterface(database):
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
            
            image_byte_array = imagePathToByteArray(image_path)

            image_name = image_path.split('/')
            image_name = image_name[-1]

            database.add_template(template_type, template_characteristic, image_name, image_byte_array)

        elif command == "2":
            template_id = input("Enter the template_id you wish to delete: ")
            database.delete_template(template_id)

        elif command == "3":
            template_id = input("Enter template_id: ")
            image = database.access_image_by_id(template_id)
            cv2.imread("template", image)

        elif command == "4":
            uploadAllTemplatesLocally(database)
        
        elif command == "5":
            id_list = database.list_of_all_IDs()
            for template_id in id_list:
                database.delete_template(template_id)

        elif command == "r":
            show_UI = False

        else:
            print("\nINCORRECT COMMAND ENTERED")

def uploadAllTemplatesLocally(database):
    if database.connected():
        template_characteristics = ["edge", "foreground"]
        template_types = ["upright", "falling", "sitting", "lying"]
        
        for template_characteristic in template_characteristics:
            for template_type in template_types:
                path = f"./templates/cropped_templates/{template_characteristic}/{template_type}/"
                for (_, _, filenames) in walk(path):
                    for filename in filenames:
                        file_path = f"{path}{filename}"
                        image = imagePathToByteArray(file_path)
                        database.add_template(template_type, template_characteristic, filename, image)
    else:
        print("Could not load files locally.")

def imagePathToByteArray(path):
    with open(path, 'rb') as f:
        byte_array = bytearray(f.read())
        return byte_array

def byteStringToImage(byteString):
    decoded = cv2.imdecode(np.frombuffer(byteString, np.uint8), -1)
    return decoded