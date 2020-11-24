import psycopg2
import numpy as np
from cv2 import cv2
from os import walk

import ComputerVision

# Database class to handle pgfunctionality
class TemplateDatabase:
    conn = None
    
    template_types = ['upright', 'falling', 'sitting', 'lying']
    template_characteristics = ['edge', 'foreground']
    template_dictionary = {'edge': {}, 'foreground': {}}

    def __init__(self, databaseName, databasePassword):
        self.database_name = databaseName
        self.database_password = databasePassword
    
    # TemplateDatabase methods
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
            template = ComputerVision.byteStringToImage(template_bytes[0].tobytes())
            curr.close()
            return template
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        
    def access_images(self, templateType, templateCharacteristic):
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
                template = ComputerVision.byteStringToImage(template_bytes)
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
            id_list = []
            for row in rows:
                id_list.append(str(row[0]))
            curr.close()
            return id_list
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def load_templates(self, templates):
        for template_characteristic in templates.keys():
            for template_type in templates[template_characteristic].keys():

                templates[template_characteristic][template_type] = \
                    self.access_images(templateType=template_type, templateCharacteristic=template_characteristic)
                    
        return templates

    def upload_all_local_templates(self):
        for template_characteristic in self.template_characteristics:
            for template_type in self.template_types:
                path = f"./templates/cropped_templates/{template_characteristic}/{template_type}/"
                for (_, _, filenames) in walk(path):
                    for filename in filenames:
                        file_path = f"{path}{filename}"
                        image = ComputerVision.imagePathToByteArray(file_path)
                        self.add_template(template_type, template_characteristic, filename, image)

# loads templates from files saved on local machine. (NOTE: Folders must be premade and organized to use)
def loadTemplatesLocally():
    local_templates = {"edge": {}, "foreground": {}}

    template_characteristics = ["edge", "foreground"]
    template_types = ["upright", "falling", "sitting", "lying"]
    
    for template_characteristic in template_characteristics:
        for template_type in template_types:
            path = f"./templates/cropped_templates/{template_characteristic}/{template_type}/"
            for (_, _, filenames) in walk(path):
                images = []
                for filename in filenames:
                    file_path = f"{path}{filename}"
                    byte_str = ComputerVision.imagePathToByteString(file_path)
                    image = ComputerVision.byteStringToImage(byte_str)
                    images.append(image)
                local_templates[template_characteristic][template_type] = images

    return local_templates
