import psycopg2
import numpy as np
from cv2 import cv2
# Database credentials 
LOCAL_DATABASE_NAME = 'CSC-450_FDS'
LOCAL_DATABASE_PASSWORD = 'Apcid28;6jdn'

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
            ''', (templateId))
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

    def load_template_dictionary(self):
        for template_type in self.template_types:
            for characteristic in self.template_characteristics:
                self.template_dictionary[characteristic][template_type] = \
                    self.access_all_image_by_type_and_chr(template_type, characteristic)
        return self.template_dictionary

def userInterface():
    database = FDSDatabase(LOCAL_DATABASE_NAME, LOCAL_DATABASE_PASSWORD)
    application_running = False
    database.connect()
    if database.connected(): 
        application_running = True
    while application_running:
        print("""
        Command:                   Description:\n
        Manage Database(m)         --   Add, Delete, or Access information stored in Database.\n
        Reload template arrays(l)  --   Loads template_type arrays (This will be automatic in future builds).\n
        Quit(q)                    --   Terminates program and disconnects from postgreSQL server.
        """)

        user_command = input("Enter a command: ")
        if user_command == "m":
            manageDatabaseMenu(database)
        elif user_command == "q":
            application_running = False
        elif user_command == "l":
            database.load_template_dictionary()
        else:
            print("\nINCORRECT COMMAND ENTERED")
    quitApplication(database)

def manageDatabaseMenu(database):
    show_manage_menu = True
    while show_manage_menu:
        print("""
        Command:        Description:\n
        1         ---   Add template.\n
        2         ---   Delete template.\n
        3         ---   Modify template.\n
        4         ---   Access template image by template_id.\n
        r         ---   Return to previous menu.
        """)
        command = input("Enter a command: ")
        if command == "1":
            print(f"{command} selected.")
            image_path = input("Please enter the path of the image: ")
            template_type = input("Please enter the template_type: ")
            template_characteristic = input("Please enter template_characteristic: ")
            verifyImagePath(image_path)
            image_name_byte_array = imagePathToByteString(image_path)
            image_name = image_name_byte_array[0]
            image_byte_array = image_name_byte_array[1]
            database.add_template(template_type, template_characteristic, image_name, image_byte_array)
        elif command == "2":
            print(f"{command} selected.")
            delete_template_id = input("Enter the template_id you wish to delete: ")
            database.delete_template(delete_template_id)
        elif command == "3":
            # TODO: Add functionality to modify template
            print(f"{command} selected.")
        elif command == "4":
            print(f"{command} selected.")
            template_id = input("Enter template_id: ")
            database.access_image_by_id(template_id)
        elif command == "r":
            show_manage_menu = False
        else:
            print(f"{command} is not a valid command.\nPlease try again.")

def imagePathToByteString(imagePath):
    file_path = imagePath.split('/')
    filename = file_path[-1]
    with open(imagePath, 'rb') as f:
        b = bytearray(f.read())
        return (filename,b)

def verifyImagePath(imagePath):
    #TODO: add functionality to verify an image.
    print("Image verified.")

def byteStringToImage(byteString):
    decoded = cv2.imdecode(np.frombuffer(byteString, np.uint8), -1)
    return decoded

def quitApplication(database):
    print("quit_application() called")
    database.disconnect()

def getImageByID(id):
    database = FDSDatabase(LOCAL_DATABASE_NAME, LOCAL_DATABASE_PASSWORD)
    database.connect()
    if database.connected():
        image = database.access_image_by_id(id)
        return image

def getAllImages():
    database = FDSDatabase(LOCAL_DATABASE_NAME, LOCAL_DATABASE_PASSWORD)
    database.connect()
    if database.connected():
        images = database.load_template_dictionary()
        return images
    else:
        return None

def main():
    userInterface()

if __name__ == '__main__':
    main()
    print("FINISHED")