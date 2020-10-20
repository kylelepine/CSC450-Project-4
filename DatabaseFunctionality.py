import psycopg2
import numpy as np
from cv2 import cv2
# Database credentials 
dbname = 'CSC-450_FDS'
pword = 'Apcid28;6jdn'

# Database class to handle pgfunctionality
class FDSDatabase:
    conn = None
    
    template_types = ['upright', 'falling', 'sitting', 'lying']
    template_characteristics = ['edge', 'foreground']
    template_dictionary = {'edge': {}, 'foreground': {}}

    def __init__(self, name, pw):
        self.dbname = name
        self.pword = pw
    
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
                database = self.dbname, user = 'postgres', \
                password = self.pword)
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

    def add_template(self, template_type, template_characteristic, image_name, image_byte_array):
        #TODO: add functionality to extract image_name from path.
        #      Maybe find last '/' and have name = what's left.
        try:
            curr = self.conn.cursor()
            print("adding template...")
            curr.execute('''
            INSERT INTO template (template_type, template_characteristic, image_name, image)
            VALUES(%s, %s, %s, %s)''', (template_type, template_characteristic, image_name, image_byte_array))
            print("Updating template table...")
            # curr.execute('''
            # UPDATE template
            # SET image = %s''', (image_byte_array + b'1',))
            self.conn.commit()
            curr.close()
            print("Added template successfully.")
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        curr.close()
        
    def delete_template(self, template_id):
        try:
            curr = self.conn.cursor()
            curr.execute('''
            DELETE FROM template
            WHERE template_id = %s
            ''', (template_id))
            self.conn.commit()
            curr.close()
            print("Deleted template successfully.")
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def access_image_by_id(self, template_id):
        try:
            curr = self.conn.cursor()
            curr.execute('''
            SELECT image
            FROM template
            WHERE template_id = %s
            ''', (template_id,))
            template_bytes = curr.fetchone()
            template = byte_str_to_image_array(template_bytes[0].tobytes())
            curr.close()
            return template
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        
    def access_all_image_by_type_and_chr(self, template_type, characteristic):
        try:
            curr = self.conn.cursor()
            curr.execute('''
            SELECT template_id, image
            FROM template
            WHERE (template_type = %s) AND (template_characteristic = %s)
            ''', (template_type, characteristic))
            rows = curr.fetchall()
            template_type_array = []
            for row in rows:
                template_bytes = row[1].tobytes()
                template = byte_str_to_image_array(template_bytes)
                template_info = (template_type, template)
                template_type_array.append(template_info)
                
            curr.close()
            print(f"Successfully loaded {characteristic} {template_type} array.")
            return template_type_array
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def load_template_dictionary(self):
        for template_type in self.template_types:
            for characteristic in self.template_characteristics:
                self.template_dictionary[characteristic][template_type] = \
                    self.access_all_image_by_type_and_chr(template_type, characteristic)
        return self.template_dictionary

# Functions
def user_interface():
    db = FDSDatabase(dbname, pword)
    application_running = False
    db.connect()
    if db.connected(): 
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
            manage_database_menu(db)
        elif user_command == "q":
            application_running = False
        elif user_command == "l":
            db.load_template_dictionary()
        else:
            print("\nINCORRECT COMMAND ENTERED")
    quit_application(db)

def manage_database_menu(database):
    print("manage_database_menu() called")
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
            verify_image_path(image_path)
            image_name_byte_array = read_image_to_byte_array(image_path)
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

def read_image_to_byte_array(image_path):
    file_path = image_path.split('/')
    filename = file_path[-1]
    with open(image_path, 'rb') as f:
        b = bytearray(f.read())
        return (filename,b)

def verify_image_path(image_path):
    #TODO: add functionality to verify an image.
    print("Image verified.")

def byte_str_to_image_array(source_str):
    decoded = cv2.imdecode(np.frombuffer(source_str, np.uint8), -1)
    return decoded

def quit_application(database):
    print("quit_application() called")
    database.disconnect()

def get_image_by_id(id):
    db = FDSDatabase(dbname, pword)
    db.connect()
    if db.connected():
        image = db.access_image_by_id(id)
        return image

def get_all_images():
    db = FDSDatabase(dbname, pword)
    db.connect()
    images = db.load_template_dictionary()
    return images

def main():
    user_interface()

if __name__ == '__main__':
    main()
    print("FINISHED")