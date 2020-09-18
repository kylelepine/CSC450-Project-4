import psycopg2


conn = None


def connect():
    global conn
    # conn parameters
    dbname = 'CSC-450_FDS'
    pword = 'Apcid28;6jdn'
    try:
        # Attempts to connect to server
        print("Connecting...")
        conn = psycopg2.connect(host = 'localhost', database = dbname, user = 'postgres', password = pword)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    
    print("Connection successful.")
    print_db_version()       


def disconnect():
    # closes communcation with PostgreSQL server
    global conn
    print("Disconnecting...")
    if conn is not None:
        conn.close()
    print("Disconnection successful.")


def print_db_version():
    print('PostgreSQL database version:')
    try:
        curr = conn.cursor()
        curr.execute('SELECT version()')
        db_version = curr.fetchone()
        curr.close()
        print(db_version)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def manage_database_menu():
    print("manage_database_menu() called")
    show_manage_menu = True
    while show_manage_menu:
        print("""
        Command:        Description:\n
        1         ---   Add template.\n
        2         ---   Delete template.\n
        3         ---   Modify template.\n
        r         ---   Return to previous menu.
        """)
        command = input("Enter a command: ")
        if command == "1":
            print(f"{command} selected.")
            image_path = input("Please enter the path of the image: ")
            template_type = input("Please enter the template_type: ")
            verify_image_path(image_path)
            image_byte_array = read_image_to_byte_array(image_path)
            add_template(template_type, image_path, image_byte_array)
        elif command == "2":
            print(f"{command} selected.")
            delete_template_id = input("Enter the template_id you wish to delete: ")
            delete_template(delete_template_id)
        elif command == "3":
            # TODO: Add functionality to modify template
            print(f"{command} selected.")
        elif command == "r":
            show_manage_menu = False
        else:
            print(f"{command} is not a valid command.\nPlease try again.")


def verify_image_path(image_path):
    #TODO: add functionality to verify an image.
    print("Image verified.")


def add_template(template_type, image_name, image_byte_array):
    #TODO: add functionality to extract image_name from path.
    #      Maybe find last '/' and have name = what's left.
    global conn
    try:
        curr = conn.cursor()
        print("adding template...")
        curr.execute('''
        INSERT INTO template (template_type, image_name, image)
        VALUES(%s, %s, %s)''', (template_type, image_name, image_byte_array))
        print("Updating template...")
        curr.execute('''
        UPDATE template
        SET image = %s''', (image_byte_array + b'1',))
        conn.commit()
        curr.close()
        print("Added template successfully.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    curr.close()
    

def delete_template(template_id):
    global conn
    try:
        curr = conn.cursor()
        curr.execute('''
        DELETE FROM template
        WHERE template_id = %s
        ''', (template_id))
        conn.commit()
        curr.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def access_template_menu():
    print("view_template_menu() called")
    show_access_template_menu = True
    while show_access_template_menu:
        print("""
        Command:        Description:\n
        1         ---   View image by template_id.\n
        r         ---   Return to previous menu.
        """)
        command = input("Enter a command: ")
        if command == '1':
            print(f"{command} selected.")
            template_id = input("Enter template_id: ")
            access_image_by_template_id(template_id)
        elif command =='r':
            show_access_template_menu = False
        else:
            print(f"{command} is not a valid command.\nPlease try again.")


def access_image_by_template_id(template_id):
    global conn
    try:
        curr = conn.cursor()
        curr.execute('''
        SELECT image
        FROM template
        WHERE template_id = %s
        ''', (template_id))
        mview = curr.fetchone()
        image = mview[0].tobytes()
        curr.close()
        return image
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def quit_application():
    print("quit_application() called")
    disconnect()


def read_image_to_byte_array(image_path):
    with open(image_path, 'rb') as f:
        b = bytearray(f.read())
        return b


def main():
    connect()
    application_running = True
    
    while application_running:
        print("""
        Command:       Description:\n
        Manage(m)    --   Add, Delete, or Modify information stored in Database.\n
        View(v)      --   View database entries.\n
        Quit(q)      --   Terminates program and disconnects from postgreSQL server.
        """)

        user_command = input("Enter a command: ")
        if user_command == "v":
            access_template_menu()
        elif user_command == "m":
            manage_database_menu()
        elif user_command == "q":
            application_running = False
        else:
            print("\nINCORRECT COMMAND ENTERED")
    
    quit_application()


if __name__ == '__main__':
    main()
    print("FINISHED")